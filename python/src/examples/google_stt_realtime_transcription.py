# https://osdn.net/users/ssor/pf/python-audiosocket/wiki/FrontPage
# Example filename: google_stt_realtime_transcription.py

import asyncio

from threading import Thread
from time import time
from queue import SimpleQueue
from socket import (
    gethostname,
    gethostbyname
)

from google.cloud import speech

import audiosocket

STREAM_TIME_LIMIT = 30.0  # 30 seconds.

STREAM_CODEC = speech.RecognitionConfig.AudioEncoding.MULAW
# `OGG_OPUS`, `WEBM_OPUS`, `SPEEX_WITH_HEADER_BYTE` and `LINEAR16` are other
# available codecs that Asterisk can encode calls in.

BIND_ADDRESS = (gethostbyname(gethostname()), 55150)

STT_CONFIG = speech.RecognitionConfig(
    encoding=STREAM_CODEC,
    sample_rate_hertz=8000,
    audio_channel_count=1,
    model="phone_call",
    language_code="en-US"
)

STT_STREAMING_CONFIG = speech.StreamingRecognitionConfig(
    config=STT_CONFIG,
    single_utterance=False,
    interim_results=True
)


class AudioSocketStream():

    def __init__(self):
        super().__init__()
        self._input_audio_queue = SimpleQueue()
        self._active_uuids = []
        self._hang_up = False

    def audio_generator(self):
        while True:
            audio = self._input_audio_queue.get()
            if len(audio) == 0:
                return
            else:
                yield audio

    def hang_up(self):
        self._hang_up = True

    def on_exception(self, uuid, peer_name, exception):
        print(f"Call with UUID {uuid} from {peer_name} caused exception:")
        print(exception)

    def on_audio(self, uuid, peer_name, audio):

        # print(f"UUID: {uuid}\nPeer name: {peer_name}\nAudio length: {len(audio)}")

        if len(audio) == 0:
            if uuid in self._active_uuids:
                print("Call " + uuid + " is over.")
                self._input_audio_queue.put(b"")
                self._active_uuids.remove(uuid)
                self._hang_up = False
            return

        if self._hang_up:
            return audiosocket.HANGUP_CALL_MESSAGE

        # To keep the examples simple, only one call is allowed to be transcribed at
        # a time. All other incoming calls are simply hung up if there is
        # already an active transcription being performed.

        if uuid not in self._active_uuids:
            if len(self._active_uuids) == 1:
                print("Received another call while already transcribing one, " \
                      + "hanging up...")
                return audiosocket.HANGUP_CALL_MESSAGE
            else:
                print(f"Received call {uuid} from peer {peer_name}, " \
                      + "beginning transcription....")
                self._active_uuids.append(uuid)

        self._input_audio_queue.put(bytes(audio))
        # Google cloud speech API expects a `bytes` object
        # (the AudioSocket callback returns a `bytearray`).


def transcribe(as_stream):
    stt_client = speech.SpeechClient()

    while True:

        print("Waiting for call to start... (say \"stop\", or simply hang up " \
              + "the call to end the transcription. There is a hard time limit of " \
              + str(STREAM_TIME_LIMIT) + " seconds, this can be changed with the " \
              + "`STREAM_TIME_LIMIT` variable.)"
              )

        for audio in as_stream.audio_generator():
            # Wait until the first packet arrives, then start sending data to
            # Google's API.
            break

        start_time = time()
        print("Call started.")

        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in as_stream.audio_generator()
        )

        responses = stt_client.streaming_recognize(STT_STREAMING_CONFIG, requests)

        print("Interim result:")
        for response in responses:

            print(response)
            # Responses typically look like the following:
            #
            # results {
            #   alternatives {
            #     transcript: "hello hello 123"
            #     confidence: 0.58563143
            #   }
            #   is_final: true
            #   result_end_time {
            #     seconds: 4
            #     nanos: 960000000
            #   }
            #   language_code: "en-us"
            # }
            # total_billed_time {
            #   seconds: 5
            # }
            # request_id: 00000000000000000

            if time() - start_time > STREAM_TIME_LIMIT:
                print("Hanging up because stream duration limit of " \
                      + str(STREAM_TIME_LIMIT) + " seconds reached."
                      )
                as_stream.hang_up()
                break

            elif "stop" in response.results[0].alternatives[0].transcript:
                print("Hanging up because the \"Stop\" keyword was detected.")
                as_stream.hang_up()
                break


async def main():
    as_stream = AudioSocketStream()

    server = await audiosocket.start_server(
        as_stream.on_audio,
        as_stream.on_exception,
        host=BIND_ADDRESS[0],
        port=BIND_ADDRESS[1]
    )

    transcription_thread = Thread(
        target=transcribe,
        args=(as_stream,)
    )

    transcription_thread.start()

    print(f"Audiosocket server listening at {BIND_ADDRESS}, expecting" \
          + " calls to be encoded in MU-LAW, this can be changed with the " \
          + "`STREAM_CODEC` variable. Only one call can be transcribed at a time.\n")

    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
