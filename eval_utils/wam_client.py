"""Websocket client for serve_wam.py (msgpack wire protocol)."""

import logging
import time

import websockets.sync.client
from openpi_client import msgpack_numpy

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class WamClientPolicy:
    """Connects to serve_wam.py and returns (24, 14) action dicts.

    Wire protocol:
        Connect → server sends msgpack metadata dict.
        Each step: client sends msgpack obs dict → server replies with msgpack action dict.

    Observation keys expected by serve_wam.py:
        observation/head_left    (H, W, 3) uint8 RGB
        observation/left_wrist   (H, W, 3) uint8 RGB
        observation/right_wrist  (H, W, 3) uint8 RGB
        observation/state        (14,) float32
        prompt                   str

    Response keys:
        actions          (24, 14) float32
        server_timing    dict
    """

    def __init__(self, host: str = "localhost", port: int = 5000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._connect()

    def _connect(self):
        logging.info("Connecting to WAM policy server at %s ...", self._uri)
        deadline = time.monotonic() + 60.0
        last_exc = None
        while time.monotonic() < deadline:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=PING_INTERVAL_SECS,
                    ping_timeout=PING_TIMEOUT_SECS,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logging.info("Connected. Server metadata: %s", metadata)
                return conn, metadata
            except Exception as e:
                last_exc = e
                time.sleep(2.0)
        raise RuntimeError(f"Could not connect to {self._uri}: {last_exc}")

    def get_server_metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        """Send observation, return action dict with key 'actions' → (24, 14) float32."""
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from WAM server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
