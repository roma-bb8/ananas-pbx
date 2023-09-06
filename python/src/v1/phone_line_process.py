from phone_line import SLIN_CHUNK_SIZE


def phone_line_process(buffers, phone_lines, uuid, audio):
    phone_line = phone_lines[uuid]
    if uuid in buffers:
        phone_line.line_out += buffers[uuid]
        del buffers[uuid]

    phone_line.line_out += audio
    if 0 != phone_line.line_in.sig_len:
        slice = phone_line.line_in.sig[phone_line.line_in.playback_slice_start:phone_line.line_in.playback_slice_end]

        phone_line.line_in.call_frame_count += 1
        phone_line.line_in.playback_slice_start += SLIN_CHUNK_SIZE
        phone_line.line_in.playback_slice_end += SLIN_CHUNK_SIZE

        if phone_line.line_in.playback_slice_start < phone_line.line_in.sig_len:
            return slice

    if 128 < len(phone_line.line_out):
        phone_line.deepgram.send(phone_line.line_out)
        phone_line.line_out = bytearray()

    return None
