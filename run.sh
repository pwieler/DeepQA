#!/bin/bash
export PATH="/home/tha/Workplace/code/anaconda2/bin:$PATH"
source activate py3-pytorch-cuda8

echo "Starting main.py: "
python -u main.py
