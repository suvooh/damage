"""Frozen inference calibration profile.

Threshold fitting and other training-time calibration code from the source
project is intentionally not part of this damage-only project. Inference reads
the pinned profile and rejects out-of-range or incomplete values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    any_damage_threshold: float
    target_recall: float
    measured_recall: float
    negative_pixel_fpr: float
    positive_pixels: int
    negative_pixels: int
    calibration_basis: str = "pixel"
    positive_cases: int = 0
    minimum_damage_coverage: float = 0.0
    exterior_floor: float = 0.5
    tile_size: int = 768
    overlap: int = 192
    horizontal_flip_tta: bool = True
    mixed_precision: bool = False
    minimum_component_pixels: int = 0
    segmentation_minimum_component_pixels: int | None = None
    segmentation_component_filter_basis: str = ""
    type_probability_multipliers: tuple[float, ...] | None = None
    type_probability_multiplier_basis: str = ""
    triage_scales: tuple[float, ...] = (1.0,)
    segmentation_scales: tuple[float, ...] | None = None
    segmentation_threshold: float | None = None
    segmentation_threshold_basis: str = ""

    def __post_init__(self) -> None:
        triage_scales = tuple(float(value) for value in self.triage_scales)
        segmentation_scales = (
            None
            if self.segmentation_scales is None
            else tuple(float(value) for value in self.segmentation_scales)
        )
        type_probability_multipliers = (
            None
            if self.type_probability_multipliers is None
            else tuple(float(value) for value in self.type_probability_multipliers)
        )
        for name, values in (
            ("triage_scales", triage_scales),
            ("segmentation_scales", segmentation_scales),
        ):
            if values is not None and (
                not values
                or len(set(values)) != len(values)
                or any(not np.isfinite(value) or value <= 0 for value in values)
            ):
                raise ValueError(f"{name} must contain unique positive finite values")
        object.__setattr__(self, "triage_scales", triage_scales)
        object.__setattr__(self, "segmentation_scales", segmentation_scales)
        object.__setattr__(
            self, "type_probability_multipliers", type_probability_multipliers
        )
        if self.minimum_component_pixels < 0:
            raise ValueError("minimum_component_pixels must be non-negative")
        if (
            self.segmentation_minimum_component_pixels is not None
            and self.segmentation_minimum_component_pixels < 0
        ):
            raise ValueError(
                "segmentation_minimum_component_pixels must be non-negative"
            )
        if type_probability_multipliers is not None and (
            not type_probability_multipliers
            or any(
                not np.isfinite(value) or value <= 0
                for value in type_probability_multipliers
            )
        ):
            raise ValueError(
                "type_probability_multipliers must contain positive finite values"
            )
        if (
            type_probability_multipliers is not None
            and not self.type_probability_multiplier_basis.strip()
        ):
            raise ValueError(
                "type_probability_multipliers requires a non-empty selection basis"
            )
        if (
            self.segmentation_minimum_component_pixels is not None
            and not self.segmentation_component_filter_basis.strip()
        ):
            raise ValueError(
                "segmentation_minimum_component_pixels requires a non-empty "
                "selection basis"
            )
        if not 0 <= self.any_damage_threshold <= 1:
            raise ValueError("any_damage_threshold must be in [0, 1]")
        if self.segmentation_threshold is not None:
            if not 0 <= self.segmentation_threshold <= 1:
                raise ValueError("segmentation_threshold must be in [0, 1]")
            if self.segmentation_threshold < self.any_damage_threshold:
                raise ValueError(
                    "segmentation_threshold must be at least the recall-first threshold"
                )
            if not self.segmentation_threshold_basis.strip():
                raise ValueError(
                    "segmentation_threshold requires a non-empty selection basis"
                )

    @property
    def effective_segmentation_threshold(self) -> float:
        """Return the precise-mask threshold, falling back for legacy profiles."""
        return (
            self.any_damage_threshold
            if self.segmentation_threshold is None
            else self.segmentation_threshold
        )

    @property
    def effective_segmentation_scales(self) -> tuple[float, ...]:
        return (
            self.triage_scales
            if self.segmentation_scales is None
            else self.segmentation_scales
        )

    @property
    def effective_segmentation_minimum_component_pixels(self) -> int:
        """Return the precise-mask filter, falling back for legacy profiles."""
        return (
            self.minimum_component_pixels
            if self.segmentation_minimum_component_pixels is None
            else self.segmentation_minimum_component_pixels
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

