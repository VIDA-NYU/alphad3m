D3M implementation from NYU - TA2
=================================

This repository is part of New York University's implementation of the `Data Driven Discovery project <https://datadrivendiscovery.org/>`__. It holds the TA2 system (pipeline generation and execution). A separate repository holds `our TA3 implementation <https://gitlab.com/ViDA-NYU/d3m/ta3>`__.

Installation
------------

You should use Docker
::

    docker build -t ta2:latest .
    # Or pull the image from GitLab
    docker pull registry.gitlab.com/vida-nyu/d3m/ta2:latest

Running on data
--------------------

With Docker, you can use the ``docker.sh`` script. Don't forget to update the paths in the script to point to the datasets on your own machine.

::

    # Search pipelines
    ./docker.sh search seed_datasets_current/185_baseball/TRAIN ta2:latest

Updating dependencies
---------------------

I am using locked versions of dependencies for reproducible builds and so everyone has the same environment. The ``requirements.txt`` contains the list of packages installed in Dockerfile.

``numpy`` and ``Cython`` need to be installed first (they are hardcoded in Dockerfile) because other packages depend on them to build.

I want to switch to Pipenv or Poetry eventually, but there are issues right now preventing me from doing it.
