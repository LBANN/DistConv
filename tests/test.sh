#!/bin/bash

export PYTORCH_MIOPEN_SUGGEST_NHWC=1 # TODO: remove when no longer needed in PyTorch

torchrun --standalone --nproc-per-node=4 -m pytest
