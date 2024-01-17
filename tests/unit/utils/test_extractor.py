import datetime
import json
import os
import pathlib
import multiprocessing
import typing

from multiprocessing.pool import ThreadPool
from unittest.mock import Mock

import pytest

from gridded_etl_tools.utils import extractor
from .test_convenience import DummyFtpClient

class DummyPool:

    def __call__(self, processes):
        self.processes = processes
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return isinstance(value, TypeError)

class TestExtractor:

    @staticmethod
    def test_pool(mocker, manager_class):

        extract = extractor.Extractor(manager_class)

        request_function = Mock()
        batch_processor = [request_function] * 3
        batch_requests = [("parameter1", "parameter2"), ("parameter1", "paramater3"), ("parameter1", "paramater4")]
        thread_count = max(1, multiprocessing.cpu_count() - 1)

        threadpool = multiprocessing.pool.ThreadPool = DummyPool()
        starmap = multiprocessing.pool.ThreadPool.starmap = Mock(autospec=True, return_value=[True])

        final_result = extract.pool(batch_processor, batch_requests)
        assert threadpool.processes == thread_count
        starmap.assert_called_with(batch_processor, batch_requests)
        assert final_result
        

    @staticmethod
    def test_pool_failed_dl(mocker, manager_class):

        extract = extractor.Extractor(manager_class)

        request_function = Mock()
        batch_processor = [request_function] * 3
        batch_requests = [("parameter1", "parameter2"), ("parameter1", "paramater3"), ("parameter1", "paramater4")]
        thread_count = max(1, multiprocessing.cpu_count() - 1)

        threadpool = multiprocessing.pool.ThreadPool = DummyPool()
        starmap = multiprocessing.pool.ThreadPool.starmap = Mock(autospec=True, return_value=[])

        final_result = extract.pool(batch_processor, batch_requests)
        assert threadpool.processes == thread_count
        starmap.assert_called_with(batch_processor, batch_requests)
        assert final_result == False

