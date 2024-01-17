import datetime
import json
import os
import pathlib
import multiprocessing
from unittest.mock import Mock

import pytest

from gridded_etl_tools.utils import extractor
from test_convenience import DummyFtpClient

class TestExtractor:

    @staticmethod
    def test_pool(mocker, manager_class):

        request_function = Mock()
        batch_processor = [request_function, request_function]
        batch_requests = [("parameter1", "parameter2"), ("parameter1", "paramater3"), ("parameter1", "paramater4")]
        thread_count = max(1, multiprocessing.cpu_count() - 1)

        starmap = Mock(extractor.starmap, return_value=True)
        threadpool = Mock(extractor.multiprocessing.pool.ThreadPool)

        extractor = extractor.Extractor(manager_class)
        final_result = extractor.pool(batch_processor, batch_requests)

        threadpool.assert_called_once_with(processes=thread_count)
        for processor, request in zip(batch_processor, batch_requests):
            starmap.assert_called_with(processor, request)
        assert request_function.call_count == 3
        assert final_result
        
