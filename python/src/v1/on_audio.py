from phone_line_process import phone_line_process
from phone_line_new import phone_line_new
from on_hangup import on_hangup


def on_audio(buffers, phone_lines, uuid, peer_name, audio):
    if uuid in phone_lines:
        if 0 == len(audio):
            return on_hangup(buffers, phone_lines, uuid)
        else:
            return phone_line_process(buffers, phone_lines, uuid, audio)
    else:
        if uuid in buffers:
            buffers[uuid] += audio
        else:
            return phone_line_new(buffers, phone_lines, uuid, audio)
    return None
