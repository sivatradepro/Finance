#!/bin/bash

python -m pip install prophet
pip install --upgrade setuptools wheel

# Install gcc
brew update
brew install gcc

# Install libssl
brew install libssl-dev

# Install libpython
# The package name might vary depending on your system. Here are some common ones:
# For Python 3.x: python3-dev or python3-devel
# For Python 2.x: python-dev or python-devel
brew install python3-dev

echo "Dependencies installed successfully."
