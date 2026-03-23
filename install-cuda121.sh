#!/usr/bin/env bash
set -e

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install cupy-cuda12x

pip install -e .
