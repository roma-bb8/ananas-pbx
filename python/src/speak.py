from tts.vocoder.inference_mel_folder import load_vocoder
from tts.inference import infer
import torch
import json

if __name__ == "__main__":
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

    sound = infer(
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
