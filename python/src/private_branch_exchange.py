from socket import (gethostname, gethostbyname)

from deepgram import Deepgram

from tts.vocoder.inference_mel_folder import load_vocoder
from tts.inference import infer

import audiosocket
import threading
import requests
import asyncio
import torch
import json
import time


########################################

def file_get_contents(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    return content


def radtts_config():
    params = {
        'config': './tts/config_ljs_dap.json',
        'radtts_path': '/usr/src/models/RADTTS-Lada.pt',
        'vocoder_path': '/usr/src/models/hifi_vocoder.pt',
        'vocoder_config_path': '/usr/src/models/hifi_config.json',
        'sigma': 0.8,
        'sigma_tkndur': 0.666,
        'sigma_f0': 1.0,
        'sigma_energy': 1.0,
        'f0_mean': 0.0,
        'f0_std': 0.0,
        'energy_mean': 0.0,
        'energy_std': 0.0,
        'token_dur_scaling': 1.0,
        'denoising_strength': 0.0,
        'n_takes': 1,
        'output_dir': './',
        'use_amp': False,
        'seed': 1234,
        'file_name': ''
    }

    with open(params['config']) as f:
        data = f.read()

    config = json.loads(data)

    params['data_config_p'] = config['data_config']
    params['model_config_p'] = config['model_config']

    return params


def transcript_received(data):
    print("transcript_received")
    if 'channel' in data:
        transcript = data['channel']['alternatives'][0]['transcript']

        if transcript:
            print('Transcript: ' + transcript + '\n')
            # self.send_openapi(transcript)
    pass


########################################


BIND_ADDRESS = (gethostbyname(gethostname()), 55150)

TOKEN_OPEN_AI = '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'
DEEPGRAM_API_KEY = '4bd940a5399d3d3758f4212481d3ad9efcb73654'

KEEP_ALIVE_SECONDS = 30
SLIN_CHUNK_SIZE = 320  # This is based on 8kHz, 20ms, 16-bit signed linear. (8kHz * 20ms * 2 bytes)

URL = 'https://api.openai.com/v1/chat/completions'
HEADERS = {
    'Authorization': 'Bearer ' + TOKEN_OPEN_AI,
    'Content-Type': 'application/json',
}

phone_lines = {}
buffers = {}
transcription = None
DEEPGRAM = Deepgram(DEEPGRAM_API_KEY)
SEND_OPENAPI = [
    {'role': 'system', 'content': file_get_contents('./staff/seller-consultant.txt')},
    {'role': 'assistant', 'content': file_get_contents('./staff/first-message.txt')}
]

# RAD_TTS_CONFIG = radtts_config()
# WEIGHTS = torch.load('/usr/src/models/RADTTS-Lada.pt', map_location='cpu')
# VOCODER, DENOISER = load_vocoder('/usr/src/models/hifi_vocoder.pt', '/usr/src/models/hifi_config.json')



########################################

class CallBuffer:
    def __init__(self):
        self.sig = bytearray(0)
        self.sig_len = 0
        self.call_frame_count = 0
        self.playback_slice_start = 0
        self.playback_slice_end = SLIN_CHUNK_SIZE


class PhoneLine:
    def __init__(self, deepgram):
        self.deepgram = deepgram
        self.line_in = CallBuffer()
        self.line_out = []
        # self.line_out = bytearray()
        pass

    def radtts_process(self, content):
        global WEIGHTS, VOCODER, DENOISER, RAD_TTS_CONFIG
        self.line_in.sig = infer(
            WEIGHTS,
            VOCODER,
            DENOISER,
            content,
            '',
            '',
            '',
            RAD_TTS_CONFIG['sigma'],
            RAD_TTS_CONFIG['sigma_tkndur'],
            RAD_TTS_CONFIG['sigma_f0'],
            RAD_TTS_CONFIG['sigma_energy'],
            RAD_TTS_CONFIG['f0_mean'],
            RAD_TTS_CONFIG['f0_std'],
            RAD_TTS_CONFIG['energy_mean'],
            RAD_TTS_CONFIG['energy_std'],
            RAD_TTS_CONFIG['token_dur_scaling'],
            RAD_TTS_CONFIG['denoising_strength'],
            RAD_TTS_CONFIG['n_takes'],
            RAD_TTS_CONFIG['output_dir'],
            RAD_TTS_CONFIG['use_amp'],
            False,
            RAD_TTS_CONFIG['seed'],
            RAD_TTS_CONFIG['file_name'],
            RAD_TTS_CONFIG['data_config_p'],
            RAD_TTS_CONFIG['model_config_p'],
        )
        self.line_in.sig_len = len(self.line_in.sig)
        pass

    def openai_received(self, data):
        if 'choices' in data:
            content = data['choices'][0]['message']['content']

            if content:
                print('Response: ' + content + '\n')
                self.radtts_process(content)
        pass

    def send_openapi(self, transcript):
        global SEND_OPENAPI, HEADERS, URL
        SEND_OPENAPI.append({'role': 'user', 'content': transcript})

        response = requests.post(URL, headers=HEADERS, data=json.dumps({
            'model': 'gpt-3.5-turbo',
            'max_tokens': 255,
            'temperature': 0,
            'messages': SEND_OPENAPI,
            'functions': [
                {
                    'name': 'create_order',
                    'description': 'Створити замовлення на доставку квітів',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string', 'description': 'Ім\'я клієнта'},
                            'order_name': {'type': 'string', 'description': 'Ім\'я отримувача'},
                            'order_number': {'type': 'string',
                                             'description': 'Мобільний номер телефону отримувача (у форматі: 099 999 99 99)'},
                            'order_address': {'type': 'string', 'description': 'Адресу отримувача (не обов\'язково)'},
                            'bouquet': {'type': 'string', 'description': 'Який букет ?'},
                            'bouquet_size': {'type': 'string', 'description': 'Який розміз букету ?'},
                            'bouquet_packaging': {'type': 'string', 'description': 'Яку упаковку ?'},
                            'bouquet_tape': {'type': 'string', 'description': 'Яку стрічку ?'},
                        },
                    },
                    'required': [
                        'name',
                        'order_name',
                        'order_number',
                        'bouquet',
                        'bouquet_size',
                        'bouquet_packaging',
                        'bouquet_tape'
                    ],
                },
            ],
            'function_call': 'auto'
        }))
        if 200 == response.status_code:
            self.openai_received(response.json())
        else:
            print('OpenAI Error: ', response.status_code)
        pass


########################################

async def phone_line_create(uuid):
    print(f"phone_line_create, uuid: {uuid}")

    global phone_lines, DEEPGRAM, transcription
    x = await DEEPGRAM.transcription.live({
        'language': 'uk',
        'model': 'general',
        'smart_format': True,
        'interim_results': False,
        'channels': 1,
        'encoding': 'linear16',
        'sample_rate': 8000
    })
    transcription = x
    transcription.register_handler(transcription.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
    transcription.register_handler(transcription.event.TRANSCRIPT_RECEIVED, transcript_received)

    phone_line = PhoneLine(transcription)
    phone_lines[uuid] = phone_line
    pass


def wrap_async_phone_line_create(uuid):
    asyncio.run(phone_line_create(uuid))
    pass


def phone_line_new(uuid, audio):
    thread = threading.Thread(target=wrap_async_phone_line_create, args=(uuid,))
    thread.daemon = True
    thread.start()

    # buffers[uuid] = bytearray()
    # buffers[uuid].buffers += audio
    buffers[uuid] = []
    buffers[uuid].append(audio)

    return None


def phone_line_process(uuid, audio):
    global phone_lines, buffers, transcription
    phone_line = phone_lines[uuid]
    if uuid in buffers:
        phone_line.line_out += buffers[uuid]
        if uuid in buffers:
            del buffers[uuid]

    phone_line.line_out.append(audio)
    # phone_line.line_out += audio
    if 0 != phone_line.line_in.sig_len:
        slice = phone_line.line_in.sig[phone_line.line_in.playback_slice_start:phone_line.line_in.playback_slice_end]

        phone_line.line_in.call_frame_count += 1
        phone_line.line_in.playback_slice_start += SLIN_CHUNK_SIZE
        phone_line.line_in.playback_slice_end += SLIN_CHUNK_SIZE

        if phone_line.line_in.playback_slice_start < phone_line.line_in.sig_len:
            return slice

    if len(phone_line.line_out) > 128:
        contents = bytearray()
        for content in phone_line.line_out:
            contents = contents + content
        transcription.send(contents)
        phone_line.line_out.clear()

    # if 2000 < len(phone_line.line_out):
    #     print('send...')
    #     phone_line.deepgram.send(phone_line.line_out)
    #     phone_line.line_out = bytearray()

    return None


async def phone_line_delete(uuid):
    global phone_lines, buffers
    deepgram = phone_lines[uuid].deepgram
    del phone_lines[uuid]
    if uuid in buffers:
        del buffers[uuid]

    print(f"Delete websocket, uuid: {uuid}")
    await deepgram.finish()
    pass


def wrap_async_phone_line_delete(uuid):
    asyncio.run(phone_line_delete(uuid))
    pass


def on_hangup(uuid):
    thread = threading.Thread(target=wrap_async_phone_line_delete, args=(uuid,))
    thread.daemon = True
    thread.start()

    return audiosocket.HANGUP_CALL_MESSAGE


def on_audio(uuid, peer_name, audio):
    global phone_lines, buffers
    if uuid in phone_lines:
        if 0 == len(audio):
            return on_hangup(uuid)
        else:
            return phone_line_process(uuid, audio)
    else:
        if uuid in buffers:
            # buffers[uuid] += audio
            buffers[uuid].append(audio)
        else:
            return phone_line_new(uuid, audio)
    return None


def on_exception(uuid, peer_name, exception):
    global phone_lines, buffers
    print(f"Call with UUID {uuid} from {peer_name} caused exception: {str(exception)}\n")
    if uuid in phone_lines:
        del phone_lines[uuid]
    if uuid in buffers:
        del buffers[uuid]
    pass


def phone_line_keep_alive():
    global phone_lines
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


async def main():
    global phone_lines, buffers

    keep_alive_websocket_thread = threading.Thread(target=phone_line_keep_alive)
    keep_alive_websocket_thread.daemon = True
    keep_alive_websocket_thread.start()

    server = await audiosocket.start_server(on_audio, on_exception, host=BIND_ADDRESS[0], port=BIND_ADDRESS[1])
    print('Private branch exchange starting...\n')
    await server.serve_forever()
    pass


if __name__ == "__main__":
    asyncio.run(main())
