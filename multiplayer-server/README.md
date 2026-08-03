# ZeroFPS prototype game server

Install and run:

```bash
python3 -m pip install websockets
python3 multiplayer-server/game_server.py --host 127.0.0.1 --port 8765
```

To print every raw message received from clients, enable debug logging:

```bash
python3 multiplayer-server/game_server.py --host 127.0.0.1 --port 8765 --debug
```

In the editor, open **Game-Server**, use `ws://127.0.0.1:8765`, choose a lobby
and unique username, then press **Connect**.

The first protocol is a small JSON WebSocket envelope. Clients send `join`, then
`publish` messages containing `lobby`, `player`, `variable`, and a JSON array of
floating-point `value`s, for example `"value":[1.0,2.0,3.0]`. The server accepts
legacy scalar values as one-element arrays. It answers with a complete `state` on join and broadcasts each
`update` to the other players in the same lobby. This Python relay is intended for
local prototyping; it has no authentication, persistence, authority, or rate limits.

Run the relay integration test with:

```bash
python3 multiplayer-server/test_server.py
```
