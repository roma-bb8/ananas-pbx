from socket import (gethostname, gethostbyname)

import audiosocket
import asyncio
from tts.vocoder.inference_mel_folder import load_vocoder
from tts.inference import infer
import torch
import json

BIND_ADDRESS = (gethostbyname(gethostname()), 55150)
SLIN_CHUNK_SIZE = 320  # This is based on 8kHz, 20ms, 16-bit signed linear. (8kHz * 20ms * 2 bytes)


class Speaker:
    def __init__(self):
        self.sig = bytearray(0)
        self.sig_len = 0
        self.call_frame_count = 0
        self.playback_slice_start = 0
        self.playback_slice_end = SLIN_CHUNK_SIZE


async def main():
    lines = {}

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

            'file_name': 'test'
        }

        with open(params['config']) as f:
            data = f.read()

        config = json.loads(data)

        params['data_config_p'] = config['data_config']
        params['model_config_p'] = config['model_config']

        return params

    VOCODER, DENOISER = load_vocoder('/usr/src/models/hifi_vocoder.pt', '/usr/src/models/hifi_config.json')
    WEIGHTS = torch.load('/usr/src/models/RADTTS-Lada.pt', map_location='cpu')
    RAD_TTS_CONFIG = radtts_config()

    def init_speak(uuid):
        lines[uuid] = {}
        speak = Speaker()
        speak.sig = infer(
            WEIGHTS,
            VOCODER,
            DENOISER,
            'Вітаю! Я - Анжеліка, продавець-консультант компанії АНАНАС. Як я можу Вам допомогти сьогодні?',
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
        lines[uuid]['speak'] = speak
        pass

    def on_audio(uuid, peer_name, audio):
        if 0 == len(audio):
            return audiosocket.HANGUP_CALL_MESSAGE

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
        pass

    init_speak('0acafcfb742f481caedf384dd27c6365')

    server = await audiosocket.start_server(
        on_audio,
        lambda uuid, peer_name, exception:
        print(f"Call with UUID {uuid} from {peer_name} caused exception: {str(exception)}."),
        host=BIND_ADDRESS[0],
        port=BIND_ADDRESS[1]
    )

    print('Private branch exchange starting...')

    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
