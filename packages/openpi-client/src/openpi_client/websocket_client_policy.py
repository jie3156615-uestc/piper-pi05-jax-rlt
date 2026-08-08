import inspect
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


def _filter_supported_kwargs(callable_obj, kwargs):
    """Keep connection options supported by the installed websockets version."""
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_timeout_s: float = 120.0,
        request_timeout_s: float = 600.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._request_timeout_s = request_timeout_s
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > self._connect_timeout_s:
                raise TimeoutError(f"Timed out waiting for policy server at {self._uri}")
            remaining = max(1.0, self._connect_timeout_s - elapsed)
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_kwargs = _filter_supported_kwargs(
                    websockets.sync.client.connect,
                    {
                        "compression": None,
                        "max_size": None,
                        "additional_headers": headers,
                        "open_timeout": min(10.0, remaining),
                        "ping_interval": None,
                        "ping_timeout": None,
                        "close_timeout": 10.0,
                    },
                )
                conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
                metadata = msgpack_numpy.unpackb(conn.recv(timeout=min(self._request_timeout_s, remaining)))
                return conn, metadata
            except (ConnectionRefusedError, OSError, TimeoutError):
                logging.info("Still waiting for server...")
                time.sleep(min(5.0, remaining))

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv(timeout=self._request_timeout_s)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
