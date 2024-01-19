import multiprocessing
import pytest
import time
import ftplib
from pathlib import PurePosixPath

from unittest.mock import Mock, patch, PropertyMock
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
    def test_pool(manager_class):
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
    def test_pool_failed_dl(manager_class):
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
        assert not final_result

    @staticmethod
    def test_pool_no_dl(manager_class):
        extract = extractor.Extractor(manager_class)

        request_function = Mock()
        batch_processor = [request_function] * 3
        batch_requests = []

        threadpool = multiprocessing.pool.ThreadPool = Mock(side_effect=DummyPool())
        starmap = multiprocessing.pool.ThreadPool.starmap = Mock(autospec=True, return_value=[])

        final_result = extract.pool(batch_processor, batch_requests)
        threadpool.assert_not_called()
        starmap.assert_not_called()
        assert not final_result


class TestS3Extractor:
    @staticmethod
    def test_s3_request(manager_class):
        extract = extractor.S3Extractor(manager_class)

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]
        kwargs = {"file_path": rfp, "scan_indices": 0, "local_file_path": lfp}

        extract.dm.kerchunkify = Mock(autospec=True)

        extract.request(*args)
        extract.dm.kerchunkify.assert_called_once_with(**kwargs)

    @staticmethod
    def test_s3_request_fail(mocker, manager_class):
        extract = extractor.S3Extractor(manager_class)

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extract.dm.kerchunkify = Mock(autospec=True, side_effect=Exception('mocked error'))
        time.sleep = Mock()  # avoid actually sleeping for large period of time

        with pytest.raises(FileNotFoundError):
            extract.request(*args)
        assert time.sleep.call_count == 5

class TestFTPExtractor:
    @staticmethod
    def test_context_manager(manager_class):
        extract = extractor.FTPExtractor(manager_class)
        ftplib.FTP = DummyFtpClient()
        host = "what a great host"

        with extract(host) as ftp:
            pass

        ftplib.FTP.login.assert_called_once()
        ftplib.FTP.close.assert_called_once()


    @staticmethod
    def test_batch_requests(manager_class):
        extract = extractor.FTPExtractor(manager_class)
        ftplib.FTP = Mock(return_value=DummyFtpClient())
        host = "what a great host"

        pattern = ".dat"

        expected_files = [PurePosixPath("two.dat"), PurePosixPath("three.dat")]
        with extract(host) as ftp:
            found_files = ftp.batch_requests(pattern)  # uses find method
        
        assert found_files == expected_files

    @staticmethod
    def test_cwd(mocker, manager_class):
        extract = extractor.FTPExtractor(manager_class)
        ftplib.FTP = DummyFtpClient()
        ftplib.FTP.pwd = Mock(return_value="")

        host = "what a great host"

        with extract(host) as ftp:
            ftp.cwd = "over there"
            ftp.cwd

        ftplib.FTP.pwd.assert_called_once()
        ftplib.FTP.cwd.assert_called_once_with("over there")

