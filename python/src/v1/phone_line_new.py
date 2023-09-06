from phone_line import PhoneLine
from deepgram import Deepgram

import threading
import asyncio

DEEPGRAM_API_KEY = '4bd940a5399d3d3758f4212481d3ad9efcb73654'


async def phone_line_create(phone_lines, uuid):
    deepgram = await Deepgram(DEEPGRAM_API_KEY).transcription.live({
        'language': 'uk',
        'model': 'general',
        'smart_format': True,
        'interim_results': False,
        'channels': 1,
        'encoding': 'linear16',
        'sample_rate': 8000
    })

    phone_line = PhoneLine(deepgram)
    phone_line.deepgram.registerHandler(deepgram.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
    phone_line.deepgram.registerHandler(deepgram.event.TRANSCRIPT_RECEIVED, phone_line.transcript_received)

    phone_lines[uuid] = phone_line
    pass


def wrap_async_phone_line_create(phone_lines, uuid):
    asyncio.run(phone_line_create(phone_lines, uuid))
    pass


def phone_line_new(buffers, phone_lines, uuid, audio):
    thread = threading.Thread(target=wrap_async_phone_line_create, args=(phone_lines, uuid))
    thread.daemon = True
    thread.start()

    buffers[uuid] = bytearray()
    buffers[uuid] += audio

    return None
