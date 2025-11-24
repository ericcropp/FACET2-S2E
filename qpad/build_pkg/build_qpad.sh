#!/bin/bash
set -e  

# Build the conda package using conda-forge channel
conda-build -c conda-forge .
