#!/bin/bash

# virtual environemnt directory name
ENV=".env"
# python version
# python3 --> 3.6 | python2 --> 2.7
PYTHON="python3"

# remove virtual environment
source scripts/remove.sh
# create virtual environment
${PYTHON} -m venv ${ENV}
# activate virtual environment
source "./${ENV}/bin/activate"
# install dependencies to virtual environment
pip install -r requirements.txt
# install qtrader package -local
pip install -e .
# run tests
source scripts/test.sh