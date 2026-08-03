#!/usr/bin/env python3
"""ZeroFPS prototype lobby relay. Requires: pip install websockets"""

import argparse
import asyncio
import json
from collections import defaultdict

from websockets.asyncio.server import serve


class Relay:
    def __init__(self, debug=False):
        self.debug = debug
        self.clients = defaultdict(dict)  # lobby -> websocket -> player
        self.values = defaultdict(lambda: defaultdict(dict))

    def debug_input(self, websocket, raw, username=None):
        if not self.debug:
            return
        peer = getattr(websocket, "remote_address", None)
        identity = username or "<joining>"
        print(f"[DEBUG] client={peer} username={identity} input={raw}", flush=True)

    async def broadcast(self, lobby, message, exclude=None):
        encoded = json.dumps(message, separators=(",", ":"))
        peers = list(self.clients[lobby])
        await asyncio.gather(
            *(peer.send(encoded) for peer in peers if peer is not exclude),
            return_exceptions=True,
        )

    async def handle(self, websocket):
        lobby = player = None
        try:
            first_raw = await websocket.recv()
            self.debug_input(websocket, first_raw)
            first = json.loads(first_raw)
            if first.get("type") != "join":
                raise ValueError("first message must be join")
            lobby = str(first.get("lobby", "default"))[:64]
            player = str(first.get("player", "player"))[:64]
            self.clients[lobby][websocket] = player
            await websocket.send(json.dumps({
                "type": "state", "lobby": lobby, "values": self.values[lobby]
            }))
            async for raw in websocket:
                self.debug_input(websocket, raw, player)
                message = json.loads(raw)
                if message.get("type") != "publish":
                    continue
                variable = str(message.get("variable", "value"))[:128]
                raw_value = message.get("value", [0.0])
                if isinstance(raw_value, list):
                    values = [float(value) for value in raw_value[:64]]
                else:
                    # Compatibility with the original scalar protocol.
                    values = [float(raw_value)]
                self.values[lobby][player][variable] = values
                await self.broadcast(lobby, {
                    "type": "update", "lobby": lobby, "player": player,
                    "variable": variable, "value": values,
                }, exclude=websocket)
        except Exception as error:
            try:
                await websocket.send(json.dumps({"type": "error", "message": str(error)}))
            except Exception:
                pass
        finally:
            if lobby is not None:
                self.clients[lobby].pop(websocket, None)
                if player is not None:
                    self.values[lobby].pop(player, None)
                    await self.broadcast(lobby, {
                        "type": "player_left", "lobby": lobby, "player": player
                    })


async def main(host, port, debug=False):
    relay = Relay(debug=debug)
    async with serve(relay.handle, host, port):
        print(f"ZeroFPS game server listening on ws://{host}:{port}")
        if debug:
            print("ZeroFPS client input debugging enabled")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print every raw message received from each client",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.host, args.port, args.debug))
    except KeyboardInterrupt:
        print("ZeroFPS game server stopped")
