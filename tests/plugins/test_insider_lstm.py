"""Tests for the optional LSTM insider telemetry preparation."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from satark.core.events import Event, EventCategory
from satark.plugins.insider.lstm import LstmInsiderDetector


def _event(hour: int, actor: str = "alice") -> Event:
    return Event(
        category=EventCategory.USB_INSERTION,
        source="test",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour),
        actor=actor,
        attributes={"count": 1},
    )


def test_lstm_sequences_are_isolated_per_actor() -> None:
    detector = LstmInsiderDetector(sequence_length=2)
    events = [_event(index) for index in range(4)] + [_event(index, "bob") for index in range(2)]

    sequences = detector._sequences(events)

    assert sequences.shape == (4, 2, 3)


def test_lstm_requires_fit_before_detection() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        LstmInsiderDetector(sequence_length=2).detect([_event(0), _event(1), _event(2)])


def test_alignment_and_zero_threshold() -> None:
    class ZeroModel:
        def predict(self, values, **kwargs):
            return np.zeros((len(values), 3))

    detector = LstmInsiderDetector(sequence_length=2)
    detector._model = ZeroModel()
    detector._mean = np.zeros(3)
    detector._scale = np.ones(3)
    detector._threshold = 0.0
    events = [_event(i) for i in range(3)]
    findings = detector.analyze(events)
    assert [f.detection.event_ids for f in findings] == [[events[1].id], [events[2].id]]
    assert all(f.score.value == pytest.approx(0.7) for f in findings)


def test_split_before_windows_and_train_only_scaling(monkeypatch) -> None:
    from types import SimpleNamespace

    from satark.plugins.insider import lstm

    captured = {}

    class Model:
        def compile(self, **kwargs):
            pass

        def fit(self, x, y, **kwargs):
            captured["training"] = x

        def predict(self, x, **kwargs):
            captured["validation"] = x
            return x[:, -1, :]

    layers = SimpleNamespace(
        **{name: lambda *args, **kwargs: None for name in ["Input", "LSTM", "Dropout", "Dense"]}
    )
    monkeypatch.setattr(
        lstm,
        "_tensorflow",
        lambda: SimpleNamespace(
            keras=SimpleNamespace(layers=layers, Sequential=lambda layers: Model())
        ),
    )
    detector = LstmInsiderDetector(sequence_length=2, validation_fraction=0.5)
    events = [_event(i).with_attribute("count", i) for i in range(8)]
    detector.fit(events)
    assert detector._mean[0] == pytest.approx(1.5)
    train_raw = captured["training"] * detector._scale + detector._mean
    validation_raw = captured["validation"] * detector._scale + detector._mean
    assert train_raw[:, :, 0].max() == pytest.approx(3)
    assert validation_raw[:, :, 0].min() == pytest.approx(4)
    assert detector.threshold == 0


def test_real_tensorflow_training_and_inference() -> None:
    pytest.importorskip("tensorflow")
    detector = LstmInsiderDetector(sequence_length=2, epochs=1, validation_fraction=0.5)
    detector.fit([_event(i) for i in range(8)])
    assert np.isfinite(detector.threshold)
    events = [_event(i + 20).with_attribute("count", 1000) for i in range(3)]
    findings = detector.analyze(events)
    assert findings
    assert findings[-1].detection.event_ids == [events[-1].id]


@pytest.mark.parametrize("count", [-1, float("nan"), float("inf")])
def test_invalid_counts_rejected(count) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        LstmInsiderDetector().fit([_event(0).with_attribute("count", count)])


def test_short_actor_baseline_rejected_before_loading_tensorflow() -> None:
    with pytest.raises(ValueError, match="BOTH"):
        LstmInsiderDetector(sequence_length=2).fit([_event(i) for i in range(4)])
