# https://osdn.net/users/ssor/pf/python-audiosocket/wiki/FrontPage
# Example filename: wave_playback.py

import asyncio
import wave
from socket import gethostname, gethostbyname

import audiosocket

BIND_ADDRESS = (gethostbyname(gethostname()), 55150)

SAMPLE_SONG = wave.open("sounds/muffin_telephone.wav", mode="r")
SAMPLE_SONG = SAMPLE_SONG.readframes(SAMPLE_SONG.getnframes())
SAMPLE_SONG_LENGTH = len(SAMPLE_SONG)


class CallState:
    def __init__(self):
        self.call_frame_count = 0
        self.playback_slice_start = 0
        self.playback_slice_end = 320


call_states = {}


def on_audio(uuid, peer_name, audio):
    if len(audio) == 0:
        print("Call with UUID " + uuid + " was hungup.")
        call_states.pop(uuid)
        return

    if uuid in call_states.keys():
        state = call_states[uuid]
    else:
        print("Received new call with UUID of " + uuid)
        state = CallState()
        call_states.update({uuid: state})

    slice = SAMPLE_SONG[state.playback_slice_start:state.playback_slice_end]

    state.call_frame_count += 1
    state.playback_slice_start += 320
    state.playback_slice_end += 320

    if state.playback_slice_start >= SAMPLE_SONG_LENGTH:
        return audiosocket.HANGUP_CALL_MESSAGE
    else:
        return slice


def on_exception(uuid, peer_name, exception):
    print(f"Call with UUID {uuid} from {peer_name} caused exception:")
    print(exception)


async def main():
    print(f"Server listening at {BIND_ADDRESS}")

    server = await audiosocket.start_server(
        on_audio,
        on_exception,
        host=BIND_ADDRESS[0],
        port=BIND_ADDRESS[1]
    )

    await server.serve_forever()


asyncio.run(main())
