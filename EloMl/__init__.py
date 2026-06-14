"""
Elo-based Machine Learning enhancement system for NHL predictions.
This module adds adaptive player and team Elo ratings with ML optimization.
"""

from .Ratings import PlayerEloSystem, TeamEloSystem, EloConfig
from .Database import EloDatabase
from .Features import EloFeatureEngine
from .MLModel import EloMLPredictor, ModelConfig
from .Updater import EloGameUpdater
from .AutoImprove import AutoImprovementEngine

__version__ = "1.0.0"

__all__ = [
    'PlayerEloSystem',
    'TeamEloSystem',
    'EloConfig',
    'EloDatabase',
    'EloFeatureEngine',
    'EloMLPredictor',
    'ModelConfig',
    'EloGameUpdater',
    'AutoImprovementEngine',
]