"""Evolutionary self-driving car project."""

from src.brains import Brain, BrainFactory, BrainRef, build_brain, register_brain
from src.car import Car
from src.config import SimulationConfig, race_track, track_variants
from src.genetic import crossover, mutate, next_generation
from src.grand_prix import Entrant, GrandPrix, Result, build_grid
from src.neural_network import NeuralNetwork
from src.obstacles import ObstacleField
from src.simulation import Simulation
from src.track import Track
from src.track_check import TrackReport, check_track

__all__ = [
    "Brain",
    "BrainFactory",
    "BrainRef",
    "Car",
    "Entrant",
    "GrandPrix",
    "Result",
    "Track",
    "ObstacleField",
    "NeuralNetwork",
    "Simulation",
    "SimulationConfig",
    "TrackReport",
    "check_track",
    "build_brain",
    "build_grid",
    "crossover",
    "mutate",
    "next_generation",
    "race_track",
    "register_brain",
    "track_variants",
]
