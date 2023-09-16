ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y wget

WORKDIR /usr/src/models

RUN wget https://github.com/egorsmkv/radtts-hifigan/releases/download/v1.0/hifi_config.json
RUN wget https://github.com/egorsmkv/radtts-istftnet/releases/download/v1.0/RADTTS-Lada.pt
RUN wget https://github.com/egorsmkv/radtts-hifigan/releases/download/v1.0/hifi_vocoder.pt

WORKDIR /usr/src/app

RUN pip install deepgram-sdk
RUN pip install requests

RUN pip install numpy
RUN pip install numba
RUN pip install scipy
RUN pip install librosa==0.9.2
RUN pip install unidecode
RUN pip install wave
RUN pip install torch

VOLUME /usr/src/app
EXPOSE 55150/tcp

ENTRYPOINT ["python", "-u", "./pbxV3.py"]
