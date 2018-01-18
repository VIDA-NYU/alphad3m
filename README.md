Data Driven Discovery implementation from New York University - TA2
===================================================================

This repository is part of NYU's implementation of the D3M project. It holds the TA2 system (pipeline generation and execution) based on [VisTrails](https://github.com/VisTrails/VisTrails). A separate location holds our TA3 implementation, based on [VisFlow](https://github.com/yubowenok/visflow).

Installation
------------

You can either use Docker:
```
docker build -t d3m_ta2_nyu .
# Or pull the image from GitLab
docker pull registry.gitlab.com/vida-nyu/d3m/ta2:latest
```

Or you can run this natively, by installing it in a Python 3 virtualenv:
```
# First get pytorch: http://pytorch.org/
pip install http:///...torch-0.3.0-post4...whl
# Then install the other requirements
pip install -r requirements.txt
# Then this package, to create the ta2_search and other commands
pip install -e .
```

Running on test data
--------------------

With Docker, you can use the `docker.sh` script. Don't forget to update the paths in the script to point to the datasets on your own machine.
```
# Train
./docker.sh ta2-test:latest ta2_search /d3m/config_train.json
# Run a test executable
./docker.sh ta2-test:latest /d3m/tmp/executables/50f99cdf-97f7-45d7-8095-35778c39093f /d3m/config_test.json
```

If you have this installed natively, you can change the paths in the `config_train.json` and `config_test.json` and use that directly:
```
# Train
ta2_search config_train.json
# Run a test executable
/tmp/d3m/executables/50f99cdf-97f7-45d7-8095-35778c39093f config_test.json
```

Updating dependencies
---------------------

I am using locked versions of dependencies for reproducible builds and so everyone has the same environment. The `requirements.txt` is generated via `pip freeze` from the dependencies listed in `requirements.in` and `setup,py`.

I want to switch to pip-tools or pipenv eventually, but there are issues right now preventing me from doing it.
