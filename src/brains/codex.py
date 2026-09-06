"""Parallel residual tanh network: Codex's racing entrant."""

from typing import Any, Mapping

import numpy as np

from src.brains import FlatGenome, Topology, register_brain


@register_brain("codex")
class CodexBrain(FlatGenome):
    PARAMS = (
        "w_reflex",
        "b_reflex",
        "w_reflex_out",
        "w_context",
        "b_context",
        "w_bend",
        "b_bend",
        "w_bend_out",
        "w_skip",
        "b_out",
    )

    def __init__(self, input_size: int, spec: Mapping[str, Any], rng: np.random.Generator) -> None:
        reflex = int(spec.get("reflex", 12))
        context = int(spec.get("context", 10))
        bend = int(spec.get("bend", 8))

        in_scale = 1.0 / np.sqrt(input_size)
        reflex_scale = 1.0 / np.sqrt(reflex)
        context_scale = 1.0 / np.sqrt(context)
        bend_scale = 1.0 / np.sqrt(bend)

        self.w_reflex = rng.normal(0.0, in_scale, size=(input_size, reflex))
        self.b_reflex = rng.normal(0.0, in_scale, size=reflex)
        self.w_reflex_out = rng.normal(0.0, reflex_scale, size=(reflex, 2))

        self.w_context = rng.normal(0.0, in_scale, size=(input_size, context))
        self.b_context = rng.normal(0.0, in_scale, size=context)
        self.w_bend = rng.normal(0.0, context_scale, size=(context, bend))
        self.b_bend = rng.normal(0.0, context_scale, size=bend)
        self.w_bend_out = rng.normal(0.0, bend_scale, size=(bend, 2))

        self.w_skip = rng.normal(0.0, in_scale, size=(input_size, 2))
        self.b_out = rng.normal(0.0, in_scale, size=2)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        reflex = np.tanh(inputs @ self.w_reflex + self.b_reflex)
        context = np.tanh(inputs @ self.w_context + self.b_context)
        bend = np.tanh(context @ self.w_bend + self.b_bend)
        drive = (
            reflex @ self.w_reflex_out
            + bend @ self.w_bend_out
            + inputs @ self.w_skip
            + self.b_out
        )
        return np.tanh(drive / np.sqrt(3.0))

    def describe(self) -> Topology:
        return Topology(
            layers=(
                ("sensors", self.w_skip.shape[0], 0),
                ("reflex", self.w_reflex.shape[1], 1),
                ("context", self.w_context.shape[1], 1),
                ("bend", self.w_bend.shape[1], 2),
                ("drive", 2, 3),
            ),
            edges=(
                (0, 1, self.w_reflex),
                (1, 4, self.w_reflex_out),
                (0, 2, self.w_context),
                (2, 3, self.w_bend),
                (3, 4, self.w_bend_out),
                (0, 4, self.w_skip),
            ),
        )

