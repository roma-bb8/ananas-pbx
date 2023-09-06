FROM debian:stable-slim as builder

ARG ASTERISK_VERSION

RUN apt-get update && \
   apt-get install -y autoconf build-essential libjansson-dev libxml2-dev libncurses5-dev libspeex-dev \
    libcurl4-openssl-dev libnewt-dev libedit-dev libspeexdsp-dev libgsm1-dev uuid-dev sqlite3 libsqlite3-dev \
    libspandsp-dev pkg-config python3-dev libssl-dev openssl libopus-dev liburiparser-dev xmlstarlet curl wget mpg123 ffmpeg uuid-runtime && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /tmp

RUN curl -o /tmp/asterisk.tar.gz https://downloads.asterisk.org/pub/telephony/asterisk/releases/asterisk-${ASTERISK_VERSION}.tar.gz && \
    tar xf /tmp/asterisk.tar.gz && \
    cd /tmp/asterisk-${ASTERISK_VERSION}

RUN curl -L -o /tmp/audiosocket.tar.gz https://github.com/CyCoreSystems/audiosocket/archive/master.tar.gz && \
   cd /tmp/asterisk-${ASTERISK_VERSION} && \
   tar xf /tmp/audiosocket.tar.gz && \
   cp audiosocket-master/asterisk/apps/* apps/ && \
   cp audiosocket-master/asterisk/channels/* channels/ && \
   cp -R audiosocket-master/asterisk/include/* include/ && \
   cp audiosocket-master/asterisk/res/* res/

# RUN curl -L -o /tmp/audiofork.tar.gz https://github.com/nadirhamid/asterisk-audiofork/archive/master.tar.gz && \
#   cd /tmp/asterisk-${ASTERISK_VERSION} && \
#   tar xf /tmp/audiofork.tar.gz && \
#   cp asterisk-audiofork-master/app_audiofork.c apps/

RUN cd /tmp/asterisk-${ASTERISK_VERSION} && \
   ./configure --with-pjproject-bundled --with-jansson-bundled --with-spandsp --with-opus && \
   make menuselect.makeopts && \
   menuselect/menuselect \
    \
    --disable BUILD_NATIVE \
    --disable CORE-SOUNDS-EN-GSM \
    --enable CORE-SOUNDS-EN-ULAW \
    --enable EXTRA-SOUNDS-EN-ULAW \
    --enable codec_opus \
    --disable chan_sip \
    \
    enuselect.makeopts && \
    make && \
    make install && \
    sh contrib/scripts/get_mp3_source.sh && \
    rm -Rf /tmp/*

FROM debian:stable-slim

COPY --from=builder /usr/sbin/asterisk /usr/sbin/
COPY --from=builder /usr/sbin/safe_asterisk /usr/sbin/
COPY --from=builder /usr/lib/libasterisk* /usr/lib/
COPY --from=builder /usr/lib/asterisk/ /usr/lib/asterisk
COPY --from=builder /var/lib/asterisk/ /var/lib/asterisk
COPY --from=builder /var/spool/asterisk/ /var/spool/asterisk

RUN apt-get update && \
   apt-get install -y gnupg libjansson4 xml2 libncurses5 libspeex1 libcurl4-openssl-dev libedit2 libspeexdsp1 \
    libgsm1 uuid libsqlite3-0 libspandsp2 libssl3 libopus0 liburiparser1 xmlstarlet curl wget mpg123 ffmpeg uuid-runtime && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

VOLUME /var/lib/asterisk /etc/asterisk
EXPOSE 5060/udp 4569/udp
WORKDIR ~

ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# core restart now
