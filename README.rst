D3M implementation from NYU - TA2
=================================

This repository is part of New York University's implementation of the `Data Driven Discovery project <https://datadrivendiscovery.org/>`__. It holds the TA2 system (pipeline generation and execution) based on `VisTrails <https://github.com/VisTrails/VisTrails>`__. A separate repository holds `our TA3 implementation <https://gitlab.com/ViDA-NYU/d3m/ta3>`__.

Installation
------------

You can either use Docker::

    docker build -t d3m_ta2_nyu .
    # Or pull the image from GitLab
    docker pull registry.gitlab.com/vida-nyu/d3m/ta2:devel

Or you can run this natively, by installing it in a Python 3 virtualenv::

    # First get build dependencies. Make sure to use versions locked in requirements.txt!
    pip install numpy==1.13.3 Cython==0.27.3
    # Then install the other requirements
    pip install -r requirements.txt
    # Then this package, to create the ta2_search and other commands
    pip install -e .

Running on test data
--------------------

With Docker, you can use the ``docker.sh`` script. Don't forget to update the paths in the script to point to the datasets on your own machine.

::

    # Generate configuration files
    ./generate_config_docker.sh /d3m/data/seed_datasets_current/185_baseball
    # Train
    ./docker.sh ta2-test:latest ta2_search /d3m/config_train.json
    # Run a test executable
    ./docker.sh ta2-test:latest /d3m/tmp/executables/50f99cdf-97f7-45d7-8095-35778c39093f /d3m/config_test.json

If you have this installed natively, you can also run it directly::

    # Generate configuration files
    ./generate_config.sh /tmp/d3m-tmp ../data/seed_datasets_current/185_baseball
    # Train
    ta2_search config_train.json
    # Run a test executable
    /tmp/d3m/executables/50f99cdf-97f7-45d7-8095-35778c39093f config_test.json

Updating dependencies
---------------------

I am using locked versions of dependencies for reproducible builds and so everyone has the same environment. The ``requirements.txt`` is generated via ``pip freeze`` from the dependencies listed in ``requirements.in`` and ``setup,py``.

``numpy`` and ``Cython`` need to be installed first (they are in Dockerfile) because other packages depend on them to build.

I want to switch to pip-tools or pipenv eventually, but there are issues right now preventing me from doing it.
