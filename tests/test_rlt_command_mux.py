from __future__ import annotations

import numpy as np

from piper_runtime.rlt_command_mux import RLTCommandMux
from piper_runtime.rlt_command_mux import TimedCommand
from piper_runtime.rlt_keyboard import RLTKeyboardStateMachine


def _cmd(value: list[float], stamp: float) -> TimedCommand:
    return TimedCommand(value=np.asarray(value, dtype=np.float32), timestamp_s=stamp)


def test_model_state_uses_fresh_model_command() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard = RLTKeyboardStateMachine().snapshot(now_s=10.0)

    result = mux.select(
        now_s=10.0,
        keyboard=keyboard,
        feedback=_cmd([0.0] * 7, 9.99),
        model=_cmd([1.0] * 7, 9.99),
        human=None,
    )

    assert result.source == "pi05"
    assert result.reason == "fresh_model"
    np.testing.assert_allclose(result.command, [1.0] * 7)


def test_model_state_holds_feedback_when_model_is_stale() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard = RLTKeyboardStateMachine().snapshot(now_s=10.0)

    result = mux.select(
        now_s=10.0,
        keyboard=keyboard,
        feedback=_cmd([0.2] * 7, 9.99),
        model=_cmd([1.0] * 7, 9.0),
        human=_cmd([2.0] * 7, 9.99),
    )

    assert result.source == "safety_block"
    assert result.reason == "model_stale"
    np.testing.assert_allclose(result.command, [0.2] * 7)


def test_first_s_only_arms_and_holds_even_with_a_fresh_human_command() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard_sm = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    keyboard = keyboard_sm.press("s", now_s=10.0)

    result = mux.select(
        now_s=10.02,
        keyboard=keyboard,
        feedback=_cmd([0.0] * 7, 10.0),
        model=_cmd([1.0] * 7, 10.0),
        human=_cmd([2.0] * 7, 10.0),
    )

    assert keyboard.mode == "TAKEOVER_ARMED"
    assert result.source == "safety_block"
    assert result.reason == "takeover_armed_hold"
    np.testing.assert_allclose(result.command, [0.0] * 7)


def test_human_command_is_selected_only_after_explicit_takeover_start() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard_sm = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    keyboard_sm.press("s", now_s=10.0)
    keyboard = keyboard_sm.start_takeover(now_s=10.02)

    result = mux.select(
        now_s=10.03,
        keyboard=keyboard,
        feedback=_cmd([0.0] * 7, 10.02),
        model=_cmd([1.0] * 7, 10.02),
        human=_cmd([2.0] * 7, 10.02),
    )

    assert keyboard.mode == "HUMAN"
    assert result.source == "human_pika"
    assert result.reason == "fresh_human"
    np.testing.assert_allclose(result.command, [2.0] * 7)


def test_human_state_never_falls_back_to_model_when_human_is_stale() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard_sm = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    keyboard_sm.press("s", now_s=10.0)
    keyboard = keyboard_sm.start_takeover(now_s=10.02)

    result = mux.select(
        now_s=10.5,
        keyboard=keyboard,
        feedback=_cmd([0.3] * 7, 10.49),
        model=_cmd([1.0] * 7, 10.49),
        human=_cmd([2.0] * 7, 10.0),
    )

    assert result.source == "safety_block"
    assert result.reason == "human_stale"
    np.testing.assert_allclose(result.command, [0.3] * 7)


def test_second_s_waits_for_reward_and_holds_feedback() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard_sm = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    keyboard_sm.press("s", now_s=10.0)
    keyboard = keyboard_sm.press("s", now_s=10.2)

    result = mux.select(
        now_s=10.21,
        keyboard=keyboard,
        feedback=_cmd([0.4] * 7, 10.20),
        model=_cmd([1.0] * 7, 10.20),
        human=_cmd([2.0] * 7, 10.20),
    )

    assert result.source == "stop"
    assert result.reason == "waiting_for_reward_hold"
    np.testing.assert_allclose(result.command, [0.4] * 7)


def test_terminal_reward_stops_and_holds_feedback() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard_sm = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    keyboard_sm.press("s", now_s=10.0)
    keyboard_sm.press("s", now_s=10.2)
    keyboard = keyboard_sm.press("1", now_s=10.3)

    result = mux.select(
        now_s=10.31,
        keyboard=keyboard,
        feedback=_cmd([0.5] * 7, 10.30),
        model=_cmd([1.0] * 7, 10.30),
        human=_cmd([2.0] * 7, 10.30),
    )

    assert result.source == "stop"
    assert result.should_stop is True
    assert result.reason == "episode_done"
    np.testing.assert_allclose(result.command, [0.5] * 7)


def test_invalid_selected_command_holds_feedback() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard = RLTKeyboardStateMachine().snapshot(now_s=10.0)

    result = mux.select(
        now_s=10.0,
        keyboard=keyboard,
        feedback=_cmd([0.6] * 7, 9.99),
        model=_cmd([float("nan")] * 7, 9.99),
        human=None,
    )

    assert result.source == "safety_block"
    assert result.reason == "model_invalid"
    np.testing.assert_allclose(result.command, [0.6] * 7)


def test_live_rlt_actor_is_selected_only_when_phase_is_active() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    keyboard = RLTKeyboardStateMachine().snapshot(now_s=10.0)

    inactive = mux.select(
        now_s=10.0,
        keyboard=keyboard,
        feedback=_cmd([0.0] * 7, 9.99),
        model=_cmd([1.0] * 7, 9.99),
        human=None,
        actor=_cmd([3.0] * 7, 9.99),
        rlt_active=False,
        allow_actor_live=True,
    )
    active = mux.select(
        now_s=10.0,
        keyboard=keyboard,
        feedback=_cmd([0.0] * 7, 9.99),
        model=_cmd([1.0] * 7, 9.99),
        human=None,
        actor=_cmd([3.0] * 7, 9.99),
        rlt_active=True,
        allow_actor_live=True,
    )

    assert inactive.source == "pi05"
    assert active.source == "rlt"
    assert active.reason == "fresh_rlt_actor"
    np.testing.assert_allclose(active.command, [3.0] * 7)


def test_live_rlt_actor_fail_open_and_human_priority() -> None:
    mux = RLTCommandMux(action_dim=7, freshness_s=0.1)
    model_keyboard = RLTKeyboardStateMachine().snapshot(now_s=10.0)
    fallback = mux.select(
        now_s=10.0,
        keyboard=model_keyboard,
        feedback=_cmd([0.0] * 7, 9.99),
        model=_cmd([1.0] * 7, 9.99),
        human=None,
        actor=_cmd([3.0] * 7, 9.0),
        rlt_active=True,
        allow_actor_live=True,
    )
    human_keyboard_sm = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    human_keyboard_sm.press("s", now_s=10.0)
    human_keyboard = human_keyboard_sm.start_takeover(now_s=10.01)
    human = mux.select(
        now_s=10.0,
        keyboard=human_keyboard,
        feedback=_cmd([0.0] * 7, 9.99),
        model=_cmd([1.0] * 7, 9.99),
        human=_cmd([2.0] * 7, 9.99),
        actor=_cmd([3.0] * 7, 9.99),
        rlt_active=True,
        allow_actor_live=True,
    )

    assert fallback.source == "pi05"
    assert fallback.reason == "rlt_actor_stale_fallback_model"
    assert human.source == "human_pika"
    np.testing.assert_allclose(human.command, [2.0] * 7)
