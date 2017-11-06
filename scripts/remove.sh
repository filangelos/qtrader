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
if [ -d ${ENV} ] ; then
    rm -rf ${ENV} ;
fi