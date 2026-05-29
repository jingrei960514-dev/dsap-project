"""
exercises/__init__.py
讓外部可以直接 from exercises import PushUp, PullUp
"""

from .base_exercise import BaseExercise, ExerciseResult, FormWarning
from .pushup        import PushUp
from .pullup        import PullUp
from .squat         import Squat

__all__ = [
    "BaseExercise",
    "ExerciseResult",
    "FormWarning",
    "PushUp",
    "PullUp",
    "Squat",
]