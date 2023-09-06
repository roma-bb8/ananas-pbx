import json
import time

KEEP_ALIVE_SECONDS = 30


def phone_line_keep_alive(phone_lines):
    while True:
        try:
            for uuid in list(phone_lines.keys()):
                if uuid in phone_lines:
                    phone_lines[uuid].deepgram.send(json.dumps({'type': 'KeepAlive'}))
                    print(f"Update keep alive websocket, uuid: {uuid}")
        except Exception as e:
            print(f"Error in keep alive websocket, message: {e}")
            continue

        time.sleep(KEEP_ALIVE_SECONDS)
    pass
