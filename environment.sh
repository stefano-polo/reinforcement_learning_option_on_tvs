#!/bin/bash

if ! python -m poetry --version &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    python -m pip install poetry
    python -m poetry config virtualenvs.in-project true
    echo "Poetry installed."
fi

python -m poetry install
python -m poetry shell
git clone -b rl_projects https://github.com/rdaluiso/baselines.git
cd ./baselines
pip install -e .
cd ..