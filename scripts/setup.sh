#!/bin/bash

# virtual environemnt directory name
ENV=".env"
# python version
# python3 --> 3.6 | python2 --> 2.7
PYTHON="python3"
# pip version
# pip3 --> python3 | pip2 --> python2
PIP="pip3"

# remove virtual environment
source scripts/remove.sh
# install `virtualenv` to global pip
${PIP} install virtualenv
# create virtual environment
virtualenv -p ${PYTHON} ${ENV}
# activate virtual environment
source "./${ENV}/bin/activate"
# install dependencies to virtual environment
pip install -r requirements.txt
# install qtrader package -local
pip install -e .
# run tests
source scripts/test.sh