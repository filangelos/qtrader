#!/bin/bash

# clean environemnt from objects and cache files
find . -name "*.pyc" -type f -delete
find . -name "__pycache__" -type d -delete
find . -name "*.ipynb*" -type d -delete