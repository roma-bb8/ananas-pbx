import asyncio

__all__ = [
    "HANGUP_CALL_MESSAGE",
    "start_server",
    "AsteriskFrameForwardError",
    "AsteriskMemoryAllocError",
    "AudioSocketReadError"
]

__version__ = 1.0

HANGUP_CALL_MESSAGE = b""
"""Can be returned from the callback function passed to `start_server` to
manually hangup the call represented by the value of the `uuid` parameter.
See documentation for `start_server()` for more information.
"""


class AsteriskFrameForwardError(Exception):
    """Raised when the AudioSocket Asterisk module is unable forward a an audio
    frame.
    """

    pass


class AsteriskMemoryAllocError(Exception):
    """Raised when the AudioSocket Asterisk module is unable to allocate memory.
    """

    pass


class AudioSocketReadError(Exception):
    """Raised when an unexpected header or payload size is received from
    Asterisk.
    """


# AudioSocket message kinds (silence and hangup are never sent by Asterisk):
_KIND_HANGUP = 0x00
_KIND_UUID = 0x01
_KIND_SILENCE = 0x02
_KIND_AUDIO = 0x10
_KIND_AUDIO_AS_BYTES = b"\x10"
_KIND_ERROR = 0xff

# AudioSocket errors that can occur on Asterisk's end (technically call hangup
# is another, but we don't want to treat it as such on our end):
_ERROR_FRAME_FORWARD = 0x02
_ERROR_MEMEORY_ALLOC = 0x04


async def start_server(on_audio_callback,
                       on_exception_callback,
                       host=None,
                       port=None,
                       **kwargs):
    """Returns an `asyncio.Server` instance using the AudioSocket protocol has
    its protocol factory. This function behaves just like
    `asyncio.start_server()`, meaning the `host`, `port` and keyword arguments
    accepted by that function have the same behavior with this one. Usually,
    `await server_forever()` is called on the returned server instance to start
    accepting connections.

    The two callback arguments are expected to have the following function
    signatures.

    `on_audio_callback`, called any time audio data from the connected peer is
    received. It is expected to be a function that accepts the following
    arguments:

      - `uuid`:      The universally unique ID that identifies this specific
                     call's audio, provided as a hexdecimal string.

      - `peer_name`: A tuple consisting of the IP address and port number of the
                     remote host the audio is being sent from.

      - `audio`:     A `bytearray` instance containing the received audio data.
                     An empty `bytearray` instance (`len(audio) == 0`)
                     indicates the call hung up and no more audio will
                     be received. Audio is either encoded in 8KHz, 16-bit
                     mono PCM (when using the standalone dialplan applcation), or
                     whatever audio codec was decided upon during call setup
                     (when using Dial() application). If this argument is empty
                     (has a length of 0), the call has been hung up and will not
                     generate any more audio.

      - Any extra keyword arguments are passed along to
        `asyncio.loop.create_server()`.

    To send audio back to Asterisk, a bytes-like object must be returned by
    this callback. Audio must be sent back in chunks of 65,536 bytes or less
    (the size must be able to fit into a 16-bit unsigned integer). This audio must
    always be encoded as 8KHz, 16-bit mono PCM, regardless of the codec in use
    for the call.

    Returning `audiosocket.HANGUP_CALL_MESSAGE` (or an empty `bytearray` instance)
    will request that the call represented by the value of the `uuid` parameter
    be hungup.

    `on_exception_callback`, called any time an exception relating to the
    connected peer is raised. It is expected to be a function that accepts the
    following arguments:

      - `uuid`:      The universally unique ID that identifies the specific
                     call which caused the exception.

      - `peer_name`: A tuple consisting of the IP address and port number of the
                     remote host the exception-causing call came from.

      - `error`:     An instance of the exception that occurred.
    """

    def factory():
        protocol = _AudioSocketProtocol(on_audio_callback, on_exception_callback)
        return protocol

    loop = asyncio.get_running_loop()
    return await loop.create_server(factory, host, port, **kwargs)


class _AudioSocketProtocol(asyncio.BufferedProtocol):

    def __init__(self, on_audio, on_exception):
        super().__init__()

        self._on_audio = on_audio
        self._on_exception = on_exception
        self._transport = None
        self._peer_name = None
        self._uuid = ""
        self._active_kind = None
        self._paused = False
        self._buffer = None
        self._write_spillover_buffer = []
        self._next_read_size = 3

    def _convey_or_raise_exception(self, exception):
        if callable(self._on_exception):
            self._on_exception(self._uuid, self._peer_name, exception)
        else:
            raise exception

    def connection_made(self, transport):
        self._peer_name = transport.get_extra_info("peername")
        self._transport = transport

    def connection_lost(self, exception):
        self._on_audio(self._uuid, self._peer_name, bytearray(0))
        if exception:
            self._convey_or_raise_exception(exception)

    def pause_writing(self):
        self._paused = True

    def resume_writing(self):
        self._paused = False

    def get_buffer(self, size_hint):
        self._buffer = bytearray(self._next_read_size)
        return self._buffer

    def buffer_updated(self, byte_count):
        if self._active_kind == None:

            if byte_count != 3:
                self._convey_or_raise_exception(AudioSocketReadError("msg"))
                return

            self._active_kind = self._buffer[0]
            # TODO: Convey error on unknown message kind?
            self._next_read_size = int.from_bytes(self._buffer[1:3], byteorder="big")
            return

        if byte_count != self._next_read_size:
            msg = "Expected {} byte payload, but received {}.".format(
                self._next_read_size,
                byte_count
            )
            self._convey_or_raise_exception(AudioSocketReadError(msg))
            self._active_kind = None

        if self._active_kind == _KIND_AUDIO:
            to_send = self._on_audio(self._uuid, self._peer_name, self._buffer)

            if to_send != None:
                if len(to_send) == 0:
                    self._transport.write(b"\x00\x00\x00")
                    self._transport.close()
                    return

                if self._paused:
                    self._write_spillover_buffer.insert(0, to_send)
                    return

                if self._write_spillover_buffer:
                    self._write_spillover_buffer.insert(0, to_send)
                    to_send = self._write_spillover_buffer.pop()

                audio_size = len(to_send).to_bytes(length=2, byteorder="big")
                to_send = _KIND_AUDIO_AS_BYTES + audio_size + to_send
                self._transport.write(to_send)

        elif self._active_kind == _KIND_UUID:
            self._uuid = self._buffer[:byte_count].hex()

        elif self._active_kind == _KIND_ERROR:
            if self._buffer[0] == _ERROR_FRAME_FORWARD:
                raise AsteriskFrameForwardError()
            elif self._buffer[0] == _ERROR_MEMEORY_ALLOC:
                raise AsteriskMemoryAllocError()

        self._active_kind = None
        self._next_read_size = 3

    def eof_received(self):
        return False
