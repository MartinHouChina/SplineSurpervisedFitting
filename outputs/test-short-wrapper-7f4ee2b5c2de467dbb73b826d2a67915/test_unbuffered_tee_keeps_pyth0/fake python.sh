#!/usr/bin/env bash
printf "unbuffered=%s\n" "$PYTHONUNBUFFERED"
printf "argument=<%s>\n" "$@"
printf "intentional failure\n" >&2
exit 7
