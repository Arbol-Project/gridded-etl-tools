import multiprocessing
import pytest
import time
import ftplib
import pathlib

from unittest.mock import Mock
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
        extract = extractor.Extractor(manager_class())

        request_function = Mock()
        batch_processor = [request_function] * 3
        batch_requests = [("parameter1", "parameter2"), ("parameter1", "paramater3"), ("parameter1", "paramater4")]
        thread_count = max(1, multiprocessing.cpu_count() - 1)

        threadpool = multiprocessing.pool.ThreadPool = DummyPool()
        starmap = multiprocessing.pool.ThreadPool.starmap = Mock(autospec=True, return_value=[True, False, True])

        final_result = extract.pool(batch_processor, batch_requests)
        assert threadpool.processes == thread_count
        starmap.assert_called_with(batch_processor, batch_requests)
        assert final_result is True

    @staticmethod
    def test_pool_failed_dl(manager_class):
        extract = extractor.Extractor(manager_class())

        request_function = Mock()
        batch_processor = [request_function] * 3
        batch_requests = [("parameter1", "parameter2"), ("parameter1", "paramater3"), ("parameter1", "paramater4")]
        thread_count = max(1, multiprocessing.cpu_count() - 1)

        threadpool = multiprocessing.pool.ThreadPool = DummyPool()
        starmap = multiprocessing.pool.ThreadPool.starmap = Mock(autospec=True, return_value=[False, False, False])

        final_result = extract.pool(batch_processor, batch_requests)
        assert threadpool.processes == thread_count
        starmap.assert_called_with(batch_processor, batch_requests)
        assert final_result is False

    @staticmethod
    def test_pool_no_dl(manager_class):
        extract = extractor.Extractor(manager_class())

        request_function = Mock()
        batch_processor = [request_function] * 3
        batch_requests = []

        threadpool = multiprocessing.pool.ThreadPool = Mock(side_effect=ValueError)
        starmap = multiprocessing.pool.ThreadPool.starmap = Mock(autospec=True, return_value=[])

        final_result = extract.pool(batch_processor, batch_requests)
        threadpool.assert_not_called()
        starmap.assert_not_called()
        assert not final_result


class TestS3Extractor:
    @staticmethod
    def test_s3_request(manager_class):
        extract = extractor.S3Extractor(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]
        kwargs = {"file_path": rfp, "scan_indices": 0, "local_file_path": lfp}

        extract.dm.kerchunkify = Mock(autospec=True)

        extract.request(*args)
        extract.dm.kerchunkify.assert_called_once_with(**kwargs)

    @staticmethod
    def test_s3_request_fail(mocker, manager_class):
        extract = extractor.S3Extractor(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extract.dm.kerchunkify = Mock(side_effect=Exception("mocked error"))
        time.sleep = Mock()  # avoid actually sleeping for large period of time

        with pytest.raises(FileNotFoundError):
            extract.request(*args)
        assert time.sleep.call_count == 5


class TestFTPExtractor:
    @staticmethod
    def test_context_manager(manager_class):
        extract = extractor.FTPExtractor(manager_class())
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.close = Mock()
        host = "what a great host"

        with extract(host) as ftp:  # noqa: F841
            pass

        assert ftp_client.contexts == 0
        ftp_client.login.assert_called_once()
        ftp_client.close.assert_called_once()

    @staticmethod
    def test_batch_requests(manager_class):
        extract = extractor.FTPExtractor(manager_class())
        host = "what a great host"

        pattern = ".dat"

        expected_files = [pathlib.PurePosixPath("two.dat"), pathlib.PurePosixPath("three.dat")]
        with extract(host) as ftp:
            found_files = ftp.batch_requests(pattern)  # uses find method

        assert found_files == expected_files

    @staticmethod
    def test_cwd(manager_class):
        extract = extractor.FTPExtractor(manager_class())
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.pwd = Mock(return_value="")

        host = "what a great host"

        with extract(host) as ftp:
            ftp.cwd = "over there"
            ftp.cwd

        ftp_client.pwd.assert_called_once()
        ftp_client.cwd.assert_called_once_with("over there")

    @staticmethod
    def test_cwd_connection_not_open(mocker, manager_class):
        """
        Test that CWD returns errors as expected if `cwd` is called when a connection
        is closed
        """
        extract = extractor.FTPExtractor(manager_class())
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.login = Mock(side_effect=ftp_client.__enter__)
        ftp_client.close = Mock(side_effect=ftp_client.__exit__)

        host = "what a great host"

        with extract(host) as ftp_session:
            ftp_session.cwd

        with pytest.raises(RuntimeError):
            ftp_session.cwd

        # TODO create a test for the cwd.setter

    @staticmethod
    def test_requests(mocker, manager_class, tmp_path):
        extract = extractor.FTPExtractor(manager_class())
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.retrbinary = Mock(side_effect=ftp_client.retrbinary)

        host = "what a great host"

        out_path = pathlib.PurePosixPath(tmp_path)
        with extract(host) as ftp:
            ftp.request(pathlib.PurePosixPath("two.dat"), out_path)

        assert ftp_client.host == host
        assert ftp_client.contexts == 0
        ftp_client.login.assert_called_once_with()
        assert ftp_client.commands == ["RETR two.dat"]
