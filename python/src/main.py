from socket import (gethostname, gethostbyname)

from tts.vocoder.inference_mel_folder import load_vocoder
from tts.inference import infer

from deepgram import Deepgram

import audiosocket
import requests
import asyncio
import struct
import torch
import copy
import json


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


########################################

BIND_ADDRESS = (gethostbyname(gethostname()), 55150)

DEEPGRAM_API_KEY = '4bd940a5399d3d3758f4212481d3ad9efcb73654'
DEEPGRAM = Deepgram(DEEPGRAM_API_KEY)

TOKEN_OPEN_AI = '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'
URL = 'https://api.openai.com/v1/chat/completions'
HEADERS = {
    'Authorization': 'Bearer ' + TOKEN_OPEN_AI,
    'Content-Type': 'application/json',
}
SEND_OPENAPI = [
    {'role': 'system', 'content': file_get_contents('./staff/seller-consultant.txt')},
    {'role': 'assistant', 'content': file_get_contents('./staff/first-message.txt')}
]

SILENCE_THRESHOLD = 1000
WAIT_SLIN = 99
SLIN_CHUNK_SIZE = 320  # This is based on 8kHz, 20ms, 16-bit signed linear. (8kHz * 20ms * 2 bytes)

VOCODER, DENOISER = load_vocoder('/usr/src/models/hifi_vocoder.pt', '/usr/src/models/hifi_config.json')
WEIGHTS = torch.load('/usr/src/models/RADTTS-Lada.pt', map_location='cpu')
RAD_TTS_CONFIG = radtts_config()


class Speaker:
    def __init__(self):
        self.sig = bytearray(0)
        self.sig_len = 0
        self.call_frame_count = 0
        self.playback_slice_start = 0
        self.playback_slice_end = SLIN_CHUNK_SIZE


async def main():
    lines = {}

    # lines = {}
    # lines[uuid] = {}
    # lines[uuid]['buffer'] = bytearray()
    # lines[uuid]['dialog'] = []
    # lines[uuid]['socket'] = obj TranscriptionLive
    # lines[uuid]['speak'] = obj Speaker
    # lines[uuid]['silence'] = int
    ########################################

    def radtts_process(uuid, content):
        speak = Speaker()
        speak.sig = infer(
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
        speak.sig_len = len(speak.sig)

        if 'speak' not in lines[uuid]:
            lines[uuid]['speak'] = speak
        pass

    ########################################

    def openai_received(uuid, data):
        if 'choices' in data:
            content = data['choices'][0]['message']['content']
            if content:
                lines[uuid]['dialog'].append({'role': 'assistant', 'content': content})
                print('Response: ' + content)
                radtts_process(uuid, content)
        pass

    def send_openapi(uuid, transcript):
        lines[uuid]['dialog'].append({'role': 'user', 'content': transcript})

        response = requests.post(URL, headers=HEADERS, data=json.dumps({
            'model': 'gpt-3.5-turbo',
            'max_tokens': 255,
            'temperature': 0,
            'messages': lines[uuid]['dialog'],
            'functions': [
                {
                    'name': 'create_order',
                    'description': 'Створити замовлення на доставку квітів',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string', 'description': 'Ім\'я клієнта'},
                            'order_name': {'type': 'string', 'description': 'Ім\'я отримувача'},
                            'order_number': {'type': 'string', 'description': 'Мобільний номер телефону отримувача (у форматі: 099 999 99 99)'},
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
            openai_received(uuid, response.json())
        else:
            print('OpenAI Error: ', response.status_code)
        pass

    ########################################

    def transcript_received(uuid, data):
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
            if transcript:
                print('Transcript: ' + transcript)
                send_openapi(uuid, transcript)
        pass

    async def init_transcription(uuid):
        if 'socket' not in lines[uuid]:
            transcription = await DEEPGRAM.transcription.live({
                'language': 'uk',
                'model': 'general',
                'smart_format': True,
                'interim_results': False,
                'channels': 1,
                'encoding': 'linear16',
                'sample_rate': 8000
            })
            transcription.register_handler(
                transcription.event.CLOSE,
                lambda c: print(f'Connection closed with code {c}.')
            )
            transcription.register_handler(
                transcription.event.TRANSCRIPT_RECEIVED,
                lambda data: transcript_received(uuid, data)
            )
            lines[uuid]['socket'] = transcription
            print(f"Create transcription, uuid: {uuid}.")

        if 'dialog' not in lines[uuid]:
            lines[uuid]['dialog'] = copy.deepcopy(SEND_OPENAPI)
        pass

    ########################################

    def on_hangup(uuid):
        if uuid not in lines:
            return audiosocket.HANGUP_CALL_MESSAGE

        if 'socket' in lines[uuid]:
            asyncio.create_task(lines[uuid]['socket'].finish())

        del lines[uuid]

        print(f"Finish transcription, uuid: {uuid}.")

        return audiosocket.HANGUP_CALL_MESSAGE

    def recognize_silence(uuid, audio):
        if 'silence' not in lines[uuid]:
            lines[uuid]['silence'] = 0

        audio_format = "<" + "h" * (len(audio) // 2)
        audio_data = struct.unpack(audio_format, audio)
        amplitude = max(audio_data)
        if amplitude < SILENCE_THRESHOLD:
            lines[uuid]['silence'] += 1
        pass

    def on_audio(uuid, peer_name, audio):
        if 0 == len(audio):
            return on_hangup(uuid)

        if uuid not in lines:
            lines[uuid] = {}
            lines[uuid]['buffer'] = bytearray()
            asyncio.create_task(init_transcription(uuid))

        if 'speak' in lines[uuid]:
            if 0 != lines[uuid]['speak'].sig_len:
                speak = lines[uuid]['speak']
                hunk = speak.sig[speak.playback_slice_start:speak.playback_slice_end]
                speak.call_frame_count += 1
                speak.playback_slice_start += SLIN_CHUNK_SIZE
                speak.playback_slice_end += SLIN_CHUNK_SIZE

                if speak.playback_slice_start < speak.sig_len:
                    return hunk
                else:
                    del lines[uuid]['speak']
                    return None

        recognize_silence(uuid, audio)
        lines[uuid]['buffer'] += audio
        if lines[uuid]['silence'] > WAIT_SLIN:
            if 'socket' in lines[uuid]:
                lines[uuid]['socket'].send(lines[uuid]['buffer'])
                lines[uuid]['buffer'] = bytearray()
                lines[uuid]['silence'] = 0
        pass

    def on_exception(uuid, peer_name, exception):
        on_hangup(uuid)
        print(f"Call with UUID {uuid} from {peer_name} caused exception: {str(exception)}.")
        pass

    ########################################

    server = await audiosocket.start_server(
        on_audio,
        on_exception,
        host=BIND_ADDRESS[0],
        port=BIND_ADDRESS[1]
    )

    print('Private branch exchange starting...')

    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
