#!/usr/bin/env python3
import asyncio
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from game_server import Relay


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_debug_mode_prints_client_identity_and_input(self):
        class Client:
            remote_address = ("127.0.0.1", 12345)

        output = StringIO()
        with redirect_stdout(output):
            Relay(debug=True).debug_input(Client(), '{"type":"publish"}', "alice")
        logged = output.getvalue()
        self.assertIn("username=alice", logged)
        self.assertIn('input={"type":"publish"}', logged)

    async def test_publish_is_broadcast_only_inside_lobby(self):
        relay = Relay()
        async with serve(relay.handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with (
                connect(f"ws://127.0.0.1:{port}") as alice,
                connect(f"ws://127.0.0.1:{port}") as bob,
            ):
                await alice.send(json.dumps({"type": "join", "lobby": "race", "player": "alice"}))
                await bob.send(json.dumps({"type": "join", "lobby": "race", "player": "bob"}))
                self.assertEqual(json.loads(await alice.recv())["type"], "state")
                self.assertEqual(json.loads(await bob.recv())["type"], "state")
                await alice.send(json.dumps({
                    "type": "publish", "lobby": "race", "player": "alice",
                    "variable": "position", "value": [1.0, 2.0, 3.0],
                }))
                update = json.loads(await asyncio.wait_for(bob.recv(), 1.0))
                self.assertEqual(update, {
                    "type": "update", "lobby": "race", "player": "alice",
                    "variable": "position", "value": [1.0, 2.0, 3.0],
                })


if __name__ == "__main__":
    unittest.main()
