from tts.vocoder.inference_mel_folder import load_vocoder
from tts.inference import infer

import requests
import torch
import json

SLIN_CHUNK_SIZE = 320  # This is based on 8kHz, 20ms, 16-bit signed linear. (8kHz * 20ms * 2 bytes)

TOKEN_OPEN_AI = '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'
URL = 'https://api.openai.com/v1/chat/completions'
HEADERS = {
    'Authorization': 'Bearer ' + TOKEN_OPEN_AI,
    'Content-Type': 'application/json',
}


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
        self.line_out = bytearray()

        self._send_openapi = [
            {'role': 'system', 'content': self.file_get_contents('../staff/seller-consultant.txt')},
            {'role': 'assistant', 'content': self.file_get_contents('../staff/first-message.txt')}
        ]
        self.rad_tts_config = self.radtts_config()
        self.weights = torch.load('/usr/src/models/RADTTS-Lada.pt', map_location='cpu')
        self.vocoder, self.denoiser = load_vocoder('/usr/src/models/hifi_vocoder.pt', '/usr/src/models/hifi_config.json')
        pass

    @staticmethod
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

    @staticmethod
    def file_get_contents(file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        return content

    def radtts_process(self, content):
        self.line_in.sig = infer(
            self.weights,
            self.vocoder,
            self.denoiser,
            content,
            '',
            '',
            '',
            self.rad_tts_config['sigma'],
            self.rad_tts_config['sigma_tkndur'],
            self.rad_tts_config['sigma_f0'],
            self.rad_tts_config['sigma_energy'],
            self.rad_tts_config['f0_mean'],
            self.rad_tts_config['f0_std'],
            self.rad_tts_config['energy_mean'],
            self.rad_tts_config['energy_std'],
            self.rad_tts_config['token_dur_scaling'],
            self.rad_tts_config['denoising_strength'],
            self.rad_tts_config['n_takes'],
            self.rad_tts_config['output_dir'],
            self.rad_tts_config['use_amp'],
            False,
            self.rad_tts_config['seed'],
            self.rad_tts_config['file_name'],
            self.rad_tts_config['data_config_p'],
            self.rad_tts_config['model_config_p'],
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
        self._send_openapi.append({'role': 'user', 'content': transcript})

        response = requests.post(URL, headers=HEADERS, data=json.dumps({
            'model': 'gpt-3.5-turbo',
            'max_tokens': 255,
            'temperature': 0,
            'messages': self._send_openapi,
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
            self.openai_received(response.json())
        else:
            print('OpenAI Error: ', response.status_code)
        pass

    def transcript_received(self, data):
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']

            if transcript:
                print('Transcript: ' + transcript + '\n')
                self.send_openapi(transcript)
        pass
