#!/bin/bash

CLAP_DIR = $1

sudo apt-get -y install python3 python3-pip python-is-python3
pip install -r requirements.txt

sudo cp optimizer.py $CLAP_DIR:/app/cli/modules
sudo cp machine-translation/machine-translation.ipynb $CLAP_DIR
sudo cp -R machine-translation/clp-config/ ~/.clap/config 

