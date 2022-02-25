"""Version of multiprocessing that doesn't use fork.

Necessary because of grpcio bug.
"""

import importlib
import logging
from multiprocessing import get_context
import os
import traceback
import sys


_mp_context = get_context('spawn')


def run_process(target, tag, **kwargs):
    """Call a Python function by name in a subprocess.

    :param target: Fully-qualified name of function to call.
    :param tag: Tag to add to logger to identify that process.
    :return: ``(proc, pipe)`` where ``proc`` is a `multiprocessing.Process`
        object and ``pipe`` is a `multiprocessing.Pipe`.
    """
    our_pipe, proc_pipe = _mp_context.Pipe()
    proc = _mp_context.Process(
        target=_invoke,
        name=tag,
        daemon=True,
        kwargs=dict(
            tag=tag,
            target=target,
            pipe=proc_pipe,
            kwargs=kwargs,
        ),
    )
    proc.start()

    return proc, our_pipe


def _invoke(tag, target, pipe, kwargs):
    """Invoked in the subprocess to setup logging and start the function.

    Arguments are read from ``sys.argv``.
    """
    tag = '{}-{}'.format(tag, os.getpid())

    logging.getLogger().handlers[:] = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:{}:%(name)s:%(message)s".format(tag),
        stream=sys.stdout)

    module, function = target.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)

    try:
        function(pipe=pipe, **kwargs)
    except Exception:
        logging.exception("Uncaught exception in subprocess %s", tag)
        error = traceback.format_exc()
        sys.stderr.write(error)
        sys.exit(1)
