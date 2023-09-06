#!/bin/sh
set -e

: ${ASTERISK_ARGS:="-fp"}
: ${CHECK_FILE:="extensions.conf"}

if [ $# -gt 0 ]; then
   exec "$@"
fi

if [ ! -e "/etc/asterisk/$CHECK_FILE" ]; then
   echo "Configuration not yet available."
   exit 1
fi

exec /usr/sbin/asterisk ${ASTERISK_ARGS}
