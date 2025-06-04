#!/usr/bin/env bash

add-apt-repository -y ppa:swi-prolog/stable #&> /dev/null
apt update #&> /dev/null
apt install -y swi-prolog
