from piper_runtime.rlt_keyboard import RLTKeyboardStateMachine
from piper_runtime.rlt_takeover_rollout import parse_scripted_keys


def test_e_ends_motion_phase_and_waits_for_reward():
    keyboard = RLTKeyboardStateMachine(toggle_debounce_s=0.25)

    keyboard.press("s", now_s=1.0)
    after_takeover = keyboard.snapshot(now_s=1.1)
    assert after_takeover.mode == "TAKEOVER_ARMED"
    assert after_takeover.human_takeover is True
    assert after_takeover.episode_done is False

    after_e = keyboard.press("e", now_s=2.0)

    assert after_e.mode == "WAIT_FOR_REWARD"
    assert after_e.human_takeover is False
    assert after_e.waiting_for_reward is True
    assert after_e.episode_done is False
    assert after_e.terminal_reward is None
    assert after_e.takeover_started_s is None
    assert after_e.takeover_ended_s is None


def test_reward_after_e_finishes_episode_with_operator_label():
    keyboard = RLTKeyboardStateMachine(toggle_debounce_s=0.25)

    keyboard.press("s", now_s=1.0)
    keyboard.press("e", now_s=2.0)
    after_reward = keyboard.press("1", now_s=3.0)

    assert after_reward.mode == "STOPPED"
    assert after_reward.episode_done is True
    assert after_reward.stop_requested is False
    assert after_reward.terminal_reward == 1.0


def test_q_remains_immediate_stop_with_failure_reward():
    keyboard = RLTKeyboardStateMachine(toggle_debounce_s=0.25)

    keyboard.press("s", now_s=1.0)
    after_q = keyboard.press("q", now_s=2.0)

    assert after_q.mode == "STOPPED"
    assert after_q.episode_done is True
    assert after_q.stop_requested is True
    assert after_q.terminal_reward == 0.0


def test_scripted_keys_accepts_q_for_emergency_stop():
    assert parse_scripted_keys("10:q") == {10: ["q"]}
