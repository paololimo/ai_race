"""Evolutionary self-driving cars: every entrant trains and races together."""

from src.brains import Brain, BrainRef, build_brain, color_of, entrants, register_brain
from src.car import Car
from src.config import SimulationConfig, race_track, track_variants
from src.genetic import crossover, mutate, next_generation
from src.simulation import Result, Simulation, Squad, Stage, format_results
from src.track import Track
from src.track_check import TrackReport, check_track

__all__ = [
    "Brain",
    "BrainRef",
    "Car",
    "Result",
    "Simulation",
    "SimulationConfig",
    "Squad",
    "Stage",
    "Track",
    "TrackReport",
    "build_brain",
    "check_track",
    "color_of",
    "crossover",
    "entrants",
    "format_results",
    "mutate",
    "next_generation",
    "race_track",
    "register_brain",
    "track_variants",
]
