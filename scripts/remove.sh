#!/bin/bash

# virtual environemnt directory name
ENV=".env"

# remove virtual environment
if [ -d ${ENV} ] ; then
    rm -rf ${ENV} ;
fi