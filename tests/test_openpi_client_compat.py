from openpi_client.websocket_client_policy import _filter_supported_kwargs


def test_filter_supported_websocket_kwargs_for_python38_client():
    def connect(uri, *, compression=None, open_timeout=None):
        return uri, compression, open_timeout

    filtered = _filter_supported_kwargs(
        connect,
        {
            "compression": None,
            "open_timeout": 10.0,
            "ping_interval": None,
            "ping_timeout": None,
        },
    )
    assert filtered == {"compression": None, "open_timeout": 10.0}
