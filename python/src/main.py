import asyncio
import json

from typing import Dict
from socket import (gethostname, gethostbyname)

import requests
from deepgram import Deepgram

import audiosocket
from tts.inference import infer
from tts.common import update_params
from tts.vocoder.inference_mel_folder import load_vocoder

import torch
import datetime

########################################

SLIN_CHUNK_SIZE = 320  # This is based on 8kHz, 20ms, 16-bit signed linear. (8kHz * 20ms * 2 bytes)
DEEPGRAM_API_KEY = '4bd940a5399d3d3758f4212481d3ad9efcb73654'
TOKEN_OPEN_AI = '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'
BIND_ADDRESS = (gethostbyname(gethostname()), 55150)

URL = 'https://api.openai.com/v1/chat/completions'
HEADERS = {
    'Authorization': 'Bearer ' + TOKEN_OPEN_AI,
    'Content-Type': 'application/json',
}


class CallState:
    def __init__(self):
        self.sig = bytearray(0)
        self.sig_len = 0
        self.call_frame_count = 0
        self.playback_slice_start = 0
        self.playback_slice_end = SLIN_CHUNK_SIZE


call_states = {}


async def main():
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
            'seed': 1234
        }
        # params['text'] = content
        # params['file_name'] = ''

        with open(params['config']) as f:
            data = f.read()

        config = json.loads(data)
        # update_params(config, params)

        params['data_config_p'] = config['data_config']
        params['model_config_p'] = config['model_config']

        return params

    rad_tts_config = radtts_config()
    weights = torch.load('/usr/src/models/RADTTS-Lada.pt', map_location='cpu')
    vocoder, denoiser = load_vocoder('/usr/src/models/hifi_vocoder.pt', '/usr/src/models/hifi_config.json')

    def radtts_process(content):
        rad_tts_config['file_name'] = 'sound_' + str(datetime.datetime.now())
        rad_tts_config['text'] = content

        state = CallState()
        state.sig = infer(
            weights,
            vocoder,
            denoiser,
            rad_tts_config['text'],
            '',
            '',
            '',
            rad_tts_config['sigma'],
            rad_tts_config['sigma_tkndur'],
            rad_tts_config['sigma_f0'],
            rad_tts_config['sigma_energy'],
            rad_tts_config['f0_mean'],
            rad_tts_config['f0_std'],
            rad_tts_config['energy_mean'],
            rad_tts_config['energy_std'],
            rad_tts_config['token_dur_scaling'],
            rad_tts_config['denoising_strength'],
            rad_tts_config['n_takes'],
            rad_tts_config['output_dir'],
            rad_tts_config['use_amp'],
            False,
            rad_tts_config['seed'],
            rad_tts_config['file_name'],
            rad_tts_config['data_config_p'],
            rad_tts_config['model_config_p'],
        )
        state.sig_len = len(state.sig)
        call_states.update({1: state})
        pass

    def file_get_contents(file_path) -> str:
        with open(file_path, 'r') as file:
            content = file.read()

        return content

    _send_openapi = [
        {'role': 'system', 'content': file_get_contents('./staff/seller-consultant.txt')},
        {'role': 'assistant', 'content': file_get_contents('./staff/first-message.txt')}
    ]

    def openai_received(data) -> None:
        if 'choices' in data:
            content = data['choices'][0]['message']['content']

            if content:
                print('Response: ' + content + '\n')
                radtts_process(content)
        pass

    def send_openapi(transcript: str):
        _send_openapi.append({'role': 'user', 'content': transcript})

        response = requests.post(URL, headers=HEADERS, data=json.dumps({
            'model': 'gpt-3.5-turbo',
            'max_tokens': 255,
            'temperature': 0,
            'messages': _send_openapi,
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
        if response.status_code == 200:
            openai_received(response.json())
        else:
            print('OpenAI Error: ', response.status_code)
        pass

    async def transcript_received(data: Dict) -> None:
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']

            if transcript:
                print('Transcript:' + transcript + '\n')
                send_openapi(transcript)
        pass

    def on_hangup(transcript, uuid, peer_name):
        print(f"Hangup call with UUID {uuid} from {peer_name}\n")
        # transcript.finish()
        pass

    _on_audio = []

    def on_audio(transcript, uuid, peer_name, audio):
        if len(audio) == 0:
            on_hangup(transcript, uuid, peer_name)
            return

        if 1 in call_states.keys():
            state = call_states[1]
            slice = state.sig[state.playback_slice_start:state.playback_slice_end]

            state.call_frame_count += 1
            state.playback_slice_start += SLIN_CHUNK_SIZE
            state.playback_slice_end += SLIN_CHUNK_SIZE

            if state.playback_slice_start >= state.sig_len:
                call_states.pop(1)
                return audiosocket.HANGUP_CALL_MESSAGE
            else:
                return slice

        _on_audio.append(audio)
        if len(_on_audio) > 128:
            contents = bytearray()
            for content in _on_audio:
                contents = contents + content
            transcript.send(contents)
            _on_audio.clear()
        pass

    deepgram = await Deepgram(DEEPGRAM_API_KEY).transcription.live({
        'language': 'uk',
        'model': 'general',
        'smart_format': True,
        'interim_results': False,
        'channels': 1,
        'encoding': 'linear16',
        'sample_rate': 8000
    })
    deepgram.registerHandler(deepgram.event.TRANSCRIPT_RECEIVED, transcript_received)
    deepgram.registerHandler(deepgram.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))

    server = await audiosocket.start_server(
        lambda uuid, peer_name, audio: on_audio(deepgram, uuid, peer_name, audio),
        lambda uuid, peer_name, exception: print(
            f"Call with UUID {uuid} from {peer_name} caused exception: {str(exception)}\n"
        ),
        host=BIND_ADDRESS[0],
        port=BIND_ADDRESS[1]
    )

    await server.serve_forever()
    print('start...\n')


if __name__ == "__main__":
    asyncio.run(main())
