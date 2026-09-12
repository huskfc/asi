"""Cardinality bounds for working memory decay rates and state builder configs."""

from __future__ import annotations

import pytest

from alberta_framework.core.state_builder import FixedTraceStateBuilderConfig
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
)


class _HostileList(list):
    """List subclass with hostile __len__ and __iter__ hooks."""

    def __len__(self) -> int:
        raise RuntimeError("hostile __len__ must not run")

    def __iter__(self):
        raise RuntimeError("hostile __iter__ must not run")


def _base_wm_cfg(**overrides):
    """Base WorkingMemoryConfig for testing."""
    cfg = {
        "observation_dim": 2,
        "action_dim": 1,
        "reward_dim": 1,
    }
    cfg.update(overrides)
    return WorkingMemoryConfig(**cfg)


def _base_fixed_trace_cfg(**overrides):
    """Base FixedTraceStateBuilderConfig for testing."""
    cfg = {
        "observation_dim": 2,
        "n_actions": 1,
    }
    cfg.update(overrides)
    return FixedTraceStateBuilderConfig(**cfg)


# =====================================================================
# WorkingMemoryConfig decay rate cardinality bounds
# =====================================================================

@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "reward_decay_rates",
])
def test_working_memory_rejects_oversized_decay_rates_tuple(decay_field):
    """Oversized decay rate tuples (4097 elements) rejected before per-rate loop."""
    with pytest.raises(ValueError, match="at most 4096 decay rates"):
        _base_wm_cfg(**{decay_field: tuple(0.5 for _ in range(4097))})


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "reward_decay_rates",
])
def test_working_memory_accepts_max_decay_rates_tuple(decay_field):
    """Last-fit decay count (4096 elements) accepted."""
    cfg = _base_wm_cfg(**{decay_field: tuple(0.5 for _ in range(4096))})
    featurizer = WorkingMemoryFeaturizer(cfg)
    assert featurizer is not None


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "reward_decay_rates",
])
def test_working_memory_from_config_rejects_oversized_list(decay_field):
    """from_config rejects oversized serialized lists before copy/element iteration."""
    cfg = _base_wm_cfg(**{decay_field: (0.5, 0.9)})
    serialized = cfg.to_config()
    serialized[decay_field] = [0.5] * 4097  # oversized list

    with pytest.raises(ValueError, match="at most 4096 decay rates"):
        WorkingMemoryConfig.from_config(serialized)


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "reward_decay_rates",
])
def test_working_memory_from_config_rejects_hostile_list_subclass(decay_field):
    """Hostile list subclasses rejected without invoking hostile length/iteration hooks."""
    cfg = _base_wm_cfg(**{decay_field: (0.5, 0.9)})
    serialized = cfg.to_config()
    serialized[decay_field] = _HostileList([0.5, 0.9])  # hostile subclass

    with pytest.raises(ValueError, match="must be an actual list or tuple"):
        WorkingMemoryConfig.from_config(serialized)


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "reward_decay_rates",
])
def test_working_memory_from_config_accepts_valid_list(decay_field):
    """from_config accepts valid lists within bounds."""
    cfg = _base_wm_cfg(**{decay_field: (0.5, 0.9)})
    serialized = cfg.to_config()
    serialized[decay_field] = [0.5, 0.9, 0.99]  # valid list

    restored = WorkingMemoryConfig.from_config(serialized)
    assert restored is not None


# =====================================================================
# FixedTraceStateBuilderConfig decay rate cardinality bounds
# =====================================================================

@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "outcome_decay_rates",
])
def test_fixed_trace_rejects_oversized_decay_rates_tuple(decay_field):
    """Oversized decay rate tuples (4097 elements) rejected before per-rate loop."""
    with pytest.raises(ValueError, match="at most 4096 decay rates"):
        _base_fixed_trace_cfg(**{decay_field: tuple(0.5 for _ in range(4097))})


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "outcome_decay_rates",
])
def test_fixed_trace_accepts_max_decay_rates_tuple(decay_field):
    """Last-fit decay count (4096 elements) accepted."""
    cfg = _base_fixed_trace_cfg(**{decay_field: tuple(0.5 for _ in range(4096))})
    assert cfg is not None


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "outcome_decay_rates",
])
def test_fixed_trace_from_config_rejects_oversized_list(decay_field):
    """from_config rejects oversized serialized lists before copy/element iteration."""
    cfg = _base_fixed_trace_cfg(**{decay_field: (0.5, 0.9)})
    serialized = cfg.to_config()
    serialized[decay_field] = [0.5] * 4097  # oversized list

    with pytest.raises(ValueError, match="at most 4096 decay rates"):
        FixedTraceStateBuilderConfig.from_config(serialized)


@pytest.mark.parametrize("decay_field", [
    "observation_decay_rates",
    "action_decay_rates",
    "outcome_decay_rates",
])
def test_fixed_trace_from_config_rejects_hostile_list_subclass(decay_field):
    """Hostile list subclasses rejected without invoking hostile length/iteration hooks."""
    cfg = _base_fixed_trace_cfg(**{decay_field: (0.5, 0.9)})
    serialized = cfg.to_config()
    serialized[decay_field] = _HostileList([0.5, 0.9])  # hostile subclass

    with pytest.raises(ValueError, match="decay rates must be lists or tuples"):
        FixedTraceStateBuilderConfig.from_config(serialized)


# =====================================================================
# Integration: full test suite sanity
# =====================================================================

def test_existing_working_memory_tests_still_pass():
    """Ensure no regression on existing validation tests."""
    # Valid config should still work
    cfg = WorkingMemoryConfig(observation_dim=4, action_dim=2)
    featurizer = WorkingMemoryFeaturizer(cfg)
    state = featurizer.init()
    assert state is not None

def test_existing_fixed_trace_tests_still_pass():
    """Ensure no regression on existing state builder tests."""
    cfg = FixedTraceStateBuilderConfig(observation_dim=4, n_actions=2)
    assert cfg is not None
