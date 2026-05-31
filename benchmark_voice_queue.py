"""
benchmark_voice_queue.py
────────────────────────
語音冷卻佇列：三種資料結構效能比較
  - dict + 線性掃描   O(n) 取最高優先
  - heapq             O(log n)
  - 分桶 deque        O(1)

執行方式：
    python benchmark_voice_queue.py

輸出：終端數字表格 + benchmark_chart.png 圖表
"""

import heapq
import time
import random
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque


# ──────────────────────────────────────────────
# 測試資料：模擬健身訓練的語音事件序列
# ──────────────────────────────────────────────

WARN_KEYS = [
    ("knee_cave",     0, 3.0),
    ("body_align",    0, 4.0),
    ("shoulder_tilt", 0, 3.0),
    ("back_lean",     1, 5.0),
    ("elbow_wide",    1, 5.0),
    ("good",          2, 1.5),
]

random.seed(42)

def make_events(n: int) -> list:
    """產生 n 個隨機語音事件（模擬 30~60fps 每幀可能觸發的警告）"""
    events, t = [], 0.0
    for _ in range(n):
        t += random.uniform(0.016, 0.033)   # 約 30~60fps 間隔
        key, pri, cd = random.choice(WARN_KEYS)
        events.append((t, key, pri, cd, key))
    return events


# ──────────────────────────────────────────────
# 實作一：dict + 線性掃描   O(n) pop
# ──────────────────────────────────────────────

class DictQueue:
    """
    最直覺的實作：把所有訊息存在 list，
    每次 pop 時線性掃描找優先級最小的。
    時間複雜度：push O(1)，pop O(n)
    """
    def __init__(self):
        self.messages   = []
        self.last_spoke = {}

    def push(self, sim_t, key, pri, cd, txt):
        if sim_t - self.last_spoke.get(key, 0) < cd:
            return
        self.messages.append((pri, sim_t, key, txt))

    def pop_best(self, sim_t):
        if not self.messages:
            return None
        # O(n) 線性掃描
        best = min(range(len(self.messages)),
                   key=lambda i: self.messages[i][0])
        item = self.messages.pop(best)
        self.last_spoke[item[2]] = sim_t
        return item

    def push_only(self, sim_t, key, pri, cd, txt):
        if sim_t - self.last_spoke.get(key, 0) < cd:
            return
        self.messages.append((pri, sim_t, key, txt))


# ──────────────────────────────────────────────
# 實作二：heapq（Min-Heap）   O(log n) pop
# ──────────────────────────────────────────────

class HeapQueue:
    """
    Python 內建 heapq：heap 自動維護最小元素在頂端，
    push/pop 都是 O(log n)。
    實際專案 voice_coach.py 使用此實作。
    """
    def __init__(self):
        self.heap       = []
        self.last_spoke = {}
        self._counter   = 0

    def push(self, sim_t, key, pri, cd, txt):
        if sim_t - self.last_spoke.get(key, 0) < cd:
            return
        heapq.heappush(self.heap, (pri, sim_t, self._counter, key, txt))
        self._counter += 1

    def pop_best(self, sim_t):
        if not self.heap:
            return None
        pri, t, _, key, txt = heapq.heappop(self.heap)
        self.last_spoke[key] = sim_t
        return (pri, t, key, txt)

    def push_only(self, sim_t, key, pri, cd, txt):
        if sim_t - self.last_spoke.get(key, 0) < cd:
            return
        heapq.heappush(self.heap, (pri, sim_t, self._counter, key, txt))
        self._counter += 1


# ──────────────────────────────────────────────
# 實作三：分桶 deque   O(1) pop
# ──────────────────────────────────────────────

class BucketDeque:
    """
    把訊息按優先級分入 3 條 deque（0=error, 1=warn, 2=good），
    pop 時從最高優先的桶取 popleft，攤銷複雜度 O(1)。
    push 也是 O(1)，理論上最快。
    """
    NUM_PRIORITIES = 3

    def __init__(self):
        self.buckets    = [deque() for _ in range(self.NUM_PRIORITIES)]
        self.last_spoke = {}

    def push(self, sim_t, key, pri, cd, txt):
        if sim_t - self.last_spoke.get(key, 0) < cd:
            return
        self.buckets[min(pri, self.NUM_PRIORITIES - 1)].append((sim_t, key, txt))

    def pop_best(self, sim_t):
        for p, bucket in enumerate(self.buckets):
            if bucket:
                t, key, txt = bucket.popleft()
                self.last_spoke[key] = sim_t
                return (p, t, key, txt)
        return None

    def push_only(self, sim_t, key, pri, cd, txt):
        if sim_t - self.last_spoke.get(key, 0) < cd:
            return
        self.buckets[min(pri, self.NUM_PRIORITIES - 1)].append((sim_t, key, txt))


# ──────────────────────────────────────────────
# Benchmark 工具函式
# ──────────────────────────────────────────────

REPEAT = 5

def bench_full(cls, events):
    """完整流程：每個 event 都 push + pop"""
    times = []
    for _ in range(REPEAT):
        obj = cls()
        t0  = time.perf_counter()
        for sim_t, key, pri, cd, txt in events:
            obj.push(sim_t, key, pri, cd, txt)
            obj.pop_best(sim_t)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000   # ms

def bench_push(cls, events):
    """只測 push 速度"""
    times = []
    for _ in range(REPEAT):
        obj = cls()
        t0  = time.perf_counter()
        for sim_t, key, pri, cd, txt in events:
            obj.push_only(sim_t, key, pri, cd, txt)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000

def bench_pop(cls, events):
    """只測 pop 速度（先填滿再清空）"""
    times = []
    for _ in range(REPEAT):
        obj = cls()
        for sim_t, key, pri, cd, txt in events:
            obj.push_only(sim_t, key, pri, cd, txt)
        last_t = events[-1][0] + 100
        t0 = time.perf_counter()
        while obj.pop_best(last_t):
            pass
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


# ──────────────────────────────────────────────
# 主程式：執行 benchmark
# ──────────────────────────────────────────────

SIZES     = [500, 1000, 2000, 5000, 10000]
IMPLS     = [
    ("dict + Linear Scan", DictQueue),
    ("heapq (Min-Heap)",   HeapQueue),
    ("Bucket deque",       BucketDeque),
]
COLORS    = ["#E05C5C", "#5B9BD5", "#70B77E"]
MARKERS   = ["o", "s", "^"]

full_data = {name: [] for name, _ in IMPLS}
push_data = {name: [] for name, _ in IMPLS}
pop_data  = {name: [] for name, _ in IMPLS}

print("=" * 65)
print("  Voice Queue Data Structure Benchmark")
print("=" * 65)

for n in SIZES:
    events = make_events(n)
    print(f"\n  Events: {n:,}")
    print(f"  {'Implementation':<22} {'Full(ms)':>10} {'Push(ms)':>10} {'Pop(ms)':>12}")
    print(f"  {'-'*56}")
    for name, cls in IMPLS:
        tf = bench_full(cls, events)
        tp = bench_push(cls, events)
        tk = bench_pop(cls, events)
        full_data[name].append(tf)
        push_data[name].append(tp)
        pop_data[name].append(tk)
        print(f"  {name:<22} {tf:>10.3f} {tp:>10.3f} {tk:>12.3f}")

# ── pop 倍數摘要 ──
print("\n" + "=" * 65)
print("  Pop Operation — Speedup vs dict at 10,000 events")
print("=" * 65)
dict_pop_10k = pop_data["dict + Linear Scan"][-1]
for name, _ in IMPLS:
    ratio = dict_pop_10k / pop_data[name][-1]
    bar   = "█" * min(int(ratio / 50), 30)
    print(f"  {name:<22}  {ratio:>6.0f}x faster  {bar}")


# ──────────────────────────────────────────────
# 繪製圖表
# ──────────────────────────────────────────────

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.facecolor":  "white",
})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Voice Queue Data Structure Benchmark  (Lower = Better)",
             fontsize=15, fontweight="bold", y=0.99)

panel_data = [
    (axes[0, 0], full_data, "Full Pipeline (push + pop per event)", False),
    (axes[0, 1], push_data, "Push Only  —  Insert Operation",        False),
    (axes[1, 0], pop_data,  "Pop Best  —  Linear Scale",             False),
    (axes[1, 1], pop_data,  "Pop Best  —  Log Scale (complexity class)", True),
]

for ax, data, title, use_log in panel_data:
    for (name, _), color, marker in zip(IMPLS, COLORS, MARKERS):
        vals = data[name]
        if use_log:
            ax.semilogy(SIZES, vals, marker=marker, color=color,
                        linewidth=2.2, markersize=7, label=name)
        else:
            ax.plot(SIZES, vals, marker=marker, color=color,
                    linewidth=2.2, markersize=7, label=name)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Time (ms)" + (" — log" if use_log else ""))
    ax.legend(fontsize=9)
    ax.set_xticks(SIZES)
    ax.tick_params(axis="x", rotation=20)

# 標注 dict 爆炸點
ax_linear = axes[1, 0]
dict_vals  = pop_data["dict + Linear Scan"]
deque_vals = pop_data["Bucket deque"]
ratio      = dict_vals[-1] / deque_vals[-1]
ax_linear.annotate(
    f"dict is {ratio:.0f}x\nslower than deque",
    xy=(10000, dict_vals[-1]),
    xytext=(6500, dict_vals[-1] * 0.6),
    fontsize=9, color="#E05C5C", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#E05C5C", lw=1.5),
)

# Big-O 標注（右下角 log 圖）
ax_log = axes[1, 1]
for (name, _), color in zip(IMPLS, COLORS):
    bigO = {"dict + Linear Scan": "O(n)",
            "heapq (Min-Heap)":   "O(log n)",
            "Bucket deque":       "O(1)"}[name]
    ax_log.annotate(bigO,
        xy=(SIZES[-1], pop_data[name][-1]),
        xytext=(SIZES[-1] + 200, pop_data[name][-1]),
        fontsize=9, color=color, fontweight="bold", va="center")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "benchmark_chart.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Chart saved → {out_path}")