#!/bin/bash

ENV=".env"
PYTHON="python3" # python3 --> 3.6 | python2 --> 2.7
PIP="pip3" # pip3 --> python3 | pip2 --> python2

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -delete

remove:
	if [ -d $(ENV) ] ; then \
		rm -rf $(ENV) ; \
	fi ;

setup:
	make remove ;
	$(PIP) install virtualenv ;
	virtualenv -p $(PYTHON) $(ENV) ;
	source "./$(ENV)/bin/activate" && \
	pip install -r requirements.txt && \
	pip install pytest && \
	make test ;

test:
	if [ ! -d $(ENV) ]; then \
		make setup ; \
	fi ;
	source "./$(ENV)/bin/activate" && \
	pytest ;
	make clean ;