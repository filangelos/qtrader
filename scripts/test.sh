#!/bin/bash

# virtual environemnt directory name
ENV=".env"

# check if virtual environment is setup
if [ ! -d ${ENV} ]
then
    # setup virtual environemnt
    source scripts/setup.sh
fi
# activate virtual environment
source "./${ENV}/bin/activate"
# run tests using `pytest`
python setup.py test
# clean environemnt
source scripts/clean.sh