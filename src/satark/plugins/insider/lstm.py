"""Optional LSTM anomaly detection for insider telemetry.

This module is intentionally separate from :class:`InsiderThreatPlugin`.
The rule-based plugin remains SATARK's default, reproducible detector; this
backend is opt-in and supplies model-derived evidence for analyst review.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from satark.core.events import Event, EventCategory
from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.evidence import Evidence, EvidenceKind
from satark.core.models.finding import Finding
from satark.core.models.score import ScoreFactor
from satark.scoring.risk import aggregate_score

_FEATURES = (
    EventCategory.USB_INSERTION,
    EventCategory.FILE_READ,
    EventCategory.FILE_WRITE,
)


@dataclass(frozen=True)
class LstmAnomaly:
    """A scored sequence ending at one actor timestamp."""

    actor: str
    event_ids: tuple[str, ...]
    timestamp: str
    score: float


class LstmInsiderDetector:
    """Per-actor LSTM reconstruction-error detector.

    Call :meth:`fit` with known-normal baseline events, then call
    :meth:`detect` on later events. Scaling and threshold calibration use only
    the baseline; the held-out tail of that baseline determines the threshold.
    TensorFlow is imported only when ``fit`` is called.
    """

    def __init__(
        self,
        *,
        sequence_length: int = 20,
        validation_fraction: float = 0.2,
        percentile: float = 99.0,
        epochs: int = 25,
        batch_size: int = 32,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be between 0 and 100")
        self.sequence_length = sequence_length
        self.validation_fraction = validation_fraction
        self.percentile = percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._threshold: float | None = None
        self._model: Any = None

    @property
    def threshold(self) -> float:
        """Calibrated reconstruction-error threshold."""
        if self._threshold is None:
            raise RuntimeError("Detector is not fitted")
        return self._threshold

    def fit(self, baseline_events: Sequence[Event]) -> None:
        """Fit on known-normal events and calibrate on a held-out tail."""
        # Split raw buckets within each actor BEFORE creating overlapping windows.
        # No observation is shared between fitting and threshold calibration.
        training_parts = []
        validation_parts = []
        for actor, buckets in self._buckets(baseline_events).items():
            vectors = np.array([row[0] for row in buckets], dtype=float)
            split = int(len(vectors) * (1.0 - self.validation_fraction))
            if min(split, len(vectors) - split) < self.sequence_length:
                raise ValueError(
                    f"Actor {actor!r} needs at least {self.sequence_length} buckets "
                    "in BOTH training and validation partitions."
                )
            training_parts.append(vectors[:split])
            validation_parts.append(vectors[split:])
        if not training_parts:
            raise ValueError("No supported baseline events")
        raw_training = np.concatenate(training_parts)
        mean = raw_training.mean(axis=0)
        scale = raw_training.std(axis=0)
        scale[scale == 0] = 1.0
        training = (np.concatenate([self._windows(p) for p in training_parts]) - mean) / scale
        validation = (np.concatenate([self._windows(p) for p in validation_parts]) - mean) / scale
        tensorflow: Any = _tensorflow()

        model = tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.Input(shape=(self.sequence_length, len(_FEATURES))),
                tensorflow.keras.layers.LSTM(64, return_sequences=True),
                tensorflow.keras.layers.Dropout(0.2),
                tensorflow.keras.layers.LSTM(32),
                tensorflow.keras.layers.Dense(len(_FEATURES)),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            training,
            training[:, -1, :],
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=0,
        )
        predicted = model.predict(validation, verbose=0)
        errors = np.mean(np.square(predicted - validation[:, -1, :]), axis=1)
        if not np.all(np.isfinite(errors)):
            raise ValueError("Model produced non-finite calibration errors")
        # Publish state only after a successful fit, including on refits.
        self._mean, self._scale = mean, scale
        self._threshold = float(np.percentile(errors, self.percentile))
        self._model = model

    def detect(self, events: Sequence[Event]) -> list[Detection]:
        """Return a detection for every sequence above the fitted threshold."""
        if self._model is None or self._mean is None or self._scale is None:
            raise RuntimeError("Call fit() before detect()")
        buckets = self._buckets(events)
        detections: list[Detection] = []
        for actor, actor_buckets in buckets.items():
            vectors = np.array([row[0] for row in actor_buckets], dtype=float)
            if len(vectors) < self.sequence_length:
                continue
            sequences = self._windows(vectors)
            scaled = self._transform(sequences)
            predicted = self._model.predict(scaled, verbose=0)
            errors = np.mean(np.square(predicted - scaled[:, -1, :]), axis=1)
            if not np.all(np.isfinite(errors)):
                raise ValueError("Model produced non-finite inference errors")
            for offset, error in enumerate(errors):
                if float(error) <= self.threshold:
                    continue
                _, timestamp, event_ids = actor_buckets[offset + self.sequence_length - 1]
                detections.append(
                    Detection(
                        plugin="insider-lstm",
                        rule_id="insider.lstm_reconstruction_error",
                        title=f"LSTM behavioral anomaly for {actor}",
                        description=(
                            "The per-actor feature sequence had reconstruction error "
                            f"{float(error):.4f}, above the calibrated threshold "
                            f"{self.threshold:.4f}."
                        ),
                        severity=DetectionSeverity.MEDIUM,
                        event_ids=[event.id for event in event_ids],
                        evidence=[
                            Evidence(
                                kind=EvidenceKind.BEHAVIORAL,
                                summary="Per-actor LSTM reconstruction anomaly",
                                details={
                                    "actor": actor,
                                    "timestamp": timestamp.isoformat(),
                                    "reconstruction_error": float(error),
                                    "threshold": self.threshold,
                                    "features": [item.value for item in _FEATURES],
                                },
                                weight=0.7,
                            )
                        ],
                        tags=["insider", "lstm", "ml", "anomaly"],
                    )
                )
        return detections

    def analyze(self, events: Sequence[Event]) -> list[Finding]:
        """Convert ML detections into transparent, reviewable SATARK findings."""
        findings: list[Finding] = []
        for detection in self.detect(events):
            evidence = detection.evidence[0]
            error = float(evidence.details["reconstruction_error"])
            factor = ScoreFactor(
                name="lstm_reconstruction_error",
                contribution=(
                    0.6 if self.threshold == 0 else min(0.6, 0.3 * error / self.threshold)
                ),
                description=(
                    f"Reconstruction error {error:.4f} exceeds the calibrated "
                    f"threshold {self.threshold:.4f}."
                ),
                evidence=list(detection.evidence),
            )
            score = aggregate_score(
                [factor],
                confidence=0.6,
                reasoning=(
                    "Optional LSTM model flagged an unusual per-actor feature sequence. "
                    "Review the listed telemetry before taking action."
                ),
                baseline=0.1,
            )
            explanation = (
                f"{detection.title}: reconstruction error {error:.4f} was above "
                f"the baseline threshold {self.threshold:.4f}."
            )
            findings.append(Finding(detection=detection, score=score, explanation=explanation))
        return findings

    def _sequences(self, events: Sequence[Event]) -> np.ndarray:
        all_sequences: list[np.ndarray] = []
        for actor_buckets in self._buckets(events).values():
            vectors = np.array([row[0] for row in actor_buckets], dtype=float)
            all_sequences.extend(
                vectors[index : index + self.sequence_length]
                for index in range(len(vectors) - self.sequence_length + 1)
            )
        return np.array(all_sequences, dtype=float)

    def _windows(self, vectors: np.ndarray) -> np.ndarray:
        return np.array(
            [
                vectors[index : index + self.sequence_length]
                for index in range(len(vectors) - self.sequence_length + 1)
            ]
        )

    @staticmethod
    def _buckets(
        events: Sequence[Event],
    ) -> dict[str, list[tuple[np.ndarray, datetime, list[Event]]]]:
        grouped: dict[str, dict[datetime, list[Event]]] = defaultdict(lambda: defaultdict(list))
        for event in events:
            if event.category in _FEATURES:
                grouped[event.actor or "unknown"][event.timestamp].append(event)
        result: dict[str, list[tuple[np.ndarray, datetime, list[Event]]]] = {}
        for actor, timestamp_events in grouped.items():
            rows: list[tuple[np.ndarray, datetime, list[Event]]] = []
            for timestamp, bucket_events in sorted(timestamp_events.items()):
                vector = np.zeros(len(_FEATURES), dtype=float)
                for event in bucket_events:
                    feature_index = _FEATURES.index(event.category)
                    count = float(event.attributes.get("count", 1))
                    if not np.isfinite(count) or count < 0:
                        raise ValueError("Activity counts must be finite and nonnegative")
                    vector[feature_index] += count
                rows.append((vector, timestamp, bucket_events))
            result[actor] = rows
        return result

    def _transform(self, values: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._scale is not None
        return (values - self._mean) / self._scale


def _tensorflow() -> object:
    """Import TensorFlow only for callers that explicitly enable ML."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "LSTM insider detection requires TensorFlow. Install it with "
            "`pip install tensorflow` before calling fit()."
        ) from exc
    return tf
