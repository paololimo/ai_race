"""Evolutionary self-driving car project."""

from src.car import Car
from src.config import SimulationConfig, race_track, track_variants
from src.genetic import crossover, mutate, next_generation
from src.neural_network import NeuralNetwork
from src.obstacles import ObstacleField
from src.simulation import Simulation
from src.track import Track
from src.track_check import TrackReport, check_track

__all__ = [
    "Car",
    "Track",
    "ObstacleField",
    "NeuralNetwork",
    "Simulation",
    "SimulationConfig",
    "TrackReport",
    "check_track",
    "crossover",
    "mutate",
    "next_generation",
    "race_track",
    "track_variants",
]
