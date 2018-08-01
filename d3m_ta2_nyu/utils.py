"""Various utilities that are not specific to D3M.
"""

import contextlib
import functools
import logging
from queue import Queue
import threading


def synchronized(lock):
    def inner(wrapped):
        def wrapper(*args, **kwargs):
            with lock:
                return wrapped(*args, **kwargs)
        functools.update_wrapper(wrapper, wrapped)
        return wrapper
    return inner


class Observable(object):
    def __init__(self):
        self.__observers = {}
        self.__next_key = 0
        self.lock = threading.RLock()

    def add_observer(self, observer):
        with self.lock:
            key = self.__next_key
            self.__next_key += 1
            self.__observers[key] = observer
            return key

    def remove_observer(self, key):
        with self.lock:
            del self.__observers[key]

    @contextlib.contextmanager
    def with_observer(self, observer):
        key = self.add_observer(observer)
        try:
            yield
        finally:
            self.remove_observer(key)

    @contextlib.contextmanager
    def with_observer_queue(self):
        queue = Queue()
        with self.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield queue

    def notify(self, event, **kwargs):
        with self.lock:
            for observer in self.__observers.values():
                try:
                    observer(event, **kwargs)
                except Exception:
                    logging.exception("Error in observer")


class ProgressStatus(object):
    def __init__(self, current, total=1.0):
        self.current = max(0.0, min(current, total))
        if total <= 0.0:
            self.total = 1.0
        else:
            self.total = total

    @property
    def ratio(self):
        return self.current / self.total

    @property
    def percentage(self):
        return '%d%%' % int(self.current / self.total)
