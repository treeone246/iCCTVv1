"""Configuration for the Windows ZeroMQ sender."""

from __future__ import annotations

# Default: WSL receiver reachable through localhost forwarding.
# If this does not work on your machine, replace with your WSL IP (example: "172.28.112.1").
ZMQ_HOST: str = "127.0.0.1"
ZMQ_PORT: int = 5555

DEFAULT_FRAME_ID: str = "camera_front"
SEND_INTERVAL_SEC: float = 0.2


def zmq_endpoint(host: str = ZMQ_HOST, port: int = ZMQ_PORT) -> str:
    """Build a ZeroMQ TCP endpoint string."""
    return f"tcp://{host}:{port}"
