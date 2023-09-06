import audiosocket
import threading
import asyncio
import queue


async def phone_line_delete(buffers, phone_lines, uuid):
    await phone_lines[uuid].deepgram.finish()

    del phone_lines[uuid]
    del buffers[uuid]
    pass


def wrap_async_phone_line_delete(buffers, phone_lines, uuid):
    asyncio.run(phone_line_delete(buffers, phone_lines, uuid))
    pass


def on_hangup(buffers, phone_lines, uuid):
    thread = threading.Thread(target=wrap_async_phone_line_delete, args=(buffers, phone_lines, uuid))
    thread.daemon = True
    thread.start()

    return audiosocket.HANGUP_CALL_MESSAGE
