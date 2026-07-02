"""Minimal websocket policy server.

A compact re-implementation of openpi's WebsocketPolicyServer (we don't have the `openpi` package in
the qwen3vl env, only `openpi_client`-compatible behaviour is needed). Wire-compatible with
`openpi_client.websocket_client_policy.WebsocketClientPolicy`: on connect it sends packed metadata, then
loops recv -> policy.infer -> attach server_timing -> send, using the vendored msgpack_numpy.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames

from . import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int | None = None, metadata: dict | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=_health_check,
        ) as server:
            logger.info("WebsocketPolicyServer listening on %s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
