"""Various utilities that are not specific to D3M.
"""

import contextlib
import logging
import json
from queue import Empty, Queue
import threading


class Observable(object):
    """Allow adding callbacks on an object, to be called on notifications.
    """
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


class _PQ_Reader(object):
    def __init__(self, pq):
        self._pq = pq
        self._pos = 0
        self.finished = False

    def get(self, timeout=None):
        if self.finished:
            return None
        with self._pq.lock:
            # There are unread items
            if (len(self._pq.list) > self._pos or
                    # Or get woken up
                    self._pq.change.wait(timeout)):
                self._pos += 1
                item = self._pq.list[self._pos - 1]
                if item is None:
                    self.finished = True
                return item
            # Timeout
            else:
                raise Empty


class PersistentQueue(object):
    """A Queue object that will always yield items inserted from the start.
    """
    def __init__(self):
        self.list = []
        self.lock = threading.RLock()
        self.change = threading.Condition(self.lock)

    def put(self, item):
        """Put an item in the queue, waking up readers.
        """
        if item is None:
            raise TypeError("Can't put None in PersistentQueue")
        with self.lock:
            self.list.append(item)
            self.change.notify_all()

    def close(self):
        """End the queue, readers will terminate.
        """
        with self.lock:
            self.list.append(None)
            self.change.notify_all()

    def read(self):
        """Get an iterator on all items from the queue.
        """
        reader = self.reader()
        while True:
            item = reader.get()
            if item is None:
                return
            yield item

    def reader(self):
        """Get a reader object you can use to read with a timeout.
        """
        return _PQ_Reader(self)


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


def is_collection(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)
        for data_resource in dataset_doc['dataResources']:
            if data_resource.get('isCollection', False):
                return True

    return False


def is_text_collection(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)
        for data_resource in dataset_doc['dataResources']:
            if data_resource.get('isCollection', False) and data_resource['resType'] == "text":
                return True

    return False
