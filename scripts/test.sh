#!/bin/bash

# virtual environemnt directory name
ENV=".env"
# python version
# python3 --> 3.6 | python2 --> 2.7
PYTHON="python3"
# pip version
# pip3 --> python3 | pip2 --> python2
PIP="pip3"

# check if virtual environment is setup
if [ ! -d ${ENV} ]
then
    # setup virtual environemnt
    source scripts/setup.sh
fi
# activate virtual environment
source "./${ENV}/bin/activate"
# run tests using `pytest`
pytest
# clean environemnt
source scripts/clean.sh