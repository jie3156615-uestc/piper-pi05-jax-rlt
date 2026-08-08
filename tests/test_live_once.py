from pathlib import Path

from piper_runtime.live_once import connect_read_only


class FakePiper:
    def __init__(self):
        self.args = None

    def ConnectPort(self, *args):
        self.args = args


def test_connect_read_only_disables_piper_initialization():
    piper = FakePiper()
    connect_read_only(piper)
    assert piper.args == (False, False, True)
    source = Path(__import__("piper_runtime.live_once").live_once.__file__).read_text()
    assert "JointCtrl" not in source
    assert "GripperCtrl" not in source
    assert "EnablePiper" not in source
