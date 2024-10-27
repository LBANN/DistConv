#!/bin/bash

torchrun --standalone --nproc-per-node=4 -m pytest
