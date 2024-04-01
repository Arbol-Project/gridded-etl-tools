import pytest
import time
import ftplib
import pathlib

from unittest.mock import Mock
from gridded_etl_tools.utils.extractor import Extractor, S3Extractor, FTPExtractor
from .test_convenience import DummyFtpClient


class ConcreteExtractor(Extractor):
    def request(*args, **kwargs):
        """
        Make base class instantiable
        """


class TestExtractor:

    @staticmethod
    def test_init():
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        assert extractor.dm == dm
        assert extractor.semaphore._value == 8

    def test_pool_request_success(self):
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        extractor.request = Mock(return_value=True)
        result = extractor.pool(batch=[{"one": 1, "two": 2}, {"one": 3, "two": 4}, {"one": 5, "two": 6}])
        assert extractor.request.call_count == 3
        assert result is True

    def test_pool_request_failure(self):
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        extractor.request = Mock(return_value=False)
        result = extractor.pool(batch=[{"one": 1, "two": 2}, {"one": 3, "two": 4}, {"one": 5, "two": 6}])
        assert extractor.request.call_count == 3
        assert result is False


class TestS3Extractor:

    @staticmethod
    def test_s3_request(manager_class):
        extractor = S3Extractor(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]
        kwargs = {"file_path": rfp, "scan_indices": 0, "local_file_path": lfp}

        extractor.dm.kerchunkify = Mock(autospec=True)

        extractor.request(*args)
        extractor.dm.kerchunkify.assert_called_once_with(**kwargs)

    @staticmethod
    def test_s3_request_with_informative_id(manager_class):
        extractor = S3Extractor(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, "informative information"]
        kwargs = {"file_path": rfp, "scan_indices": 0, "local_file_path": lfp}

        extractor.dm.kerchunkify = Mock(autospec=True)

        extractor.request(*args)
        extractor.dm.kerchunkify.assert_called_once_with(**kwargs)

    @staticmethod
    def test_s3_request_remote_file_is_not_on_s3(manager_class):
        extractor = S3Extractor(manager_class())

        rfp = "t4://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extractor.dm.kerchunkify = Mock(autospec=True)

        with pytest.raises(ValueError):
            extractor.request(*args)

        extractor.dm.kerchunkify.assert_not_called()

    @staticmethod
    def test_s3_request_fail(manager_class):
        extract = S3Extractor(manager_class())

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
    def test_context_manager():
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.close = Mock()
        host = "what a great host"

        with FTPExtractor(dm, host):
            pass

        assert ftp_client.contexts == 0
        ftp_client.login.assert_called_once()
        ftp_client.close.assert_called_once()

    @staticmethod
    def test_context_manager_no_host():
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.close = Mock()

        with pytest.raises(TypeError):
            with FTPExtractor(dm):
                pass  # pragma NO COVER

        assert ftp_client.contexts == 0
        ftp_client.login.assert_not_called()
        ftp_client.close.assert_not_called()

    @staticmethod
    def test_batch_requests():
        dm = Mock()
        host = "what a great host"

        pattern = ".dat"

        expected_files = [pathlib.PurePosixPath("two.dat"), pathlib.PurePosixPath("three.dat")]
        with FTPExtractor(dm, host) as ftp:
            found_files = ftp.batch_requests(pattern)  # uses find method

        assert found_files == expected_files

    @staticmethod
    def test_cwd():
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.pwd = Mock(return_value="")

        host = "what a great host"

        with FTPExtractor(dm, host) as ftp:
            assert ftp.cwd == pathlib.PosixPath("")
            ftp.cwd = "over there"

        ftp_client.pwd.assert_called_once()
        ftp_client.cwd.assert_called_once_with("over there")

    @staticmethod
    def test_cwd_setter_no_such_path():
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.pwd = Mock(return_value="")
        ftp_client.nlst = Mock(return_value=None)

        host = "what a great host"

        with FTPExtractor(dm, host) as ftp:
            with pytest.raises(RuntimeError):
                ftp.cwd = "over there"

        ftp_client.pwd.assert_not_called()
        ftp_client.cwd.assert_not_called()
        ftp_client.nlst.assert_called_once_with("over there")

    @staticmethod
    def test_cwd_client_error():
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.pwd = Mock(return_value="")
        ftp_client.cwd = Mock(side_effect=ftplib.error_perm)

        host = "what a great host"

        with FTPExtractor(dm, host) as ftp:
            with pytest.raises(RuntimeError):
                ftp.cwd = "over there"

        ftp_client.pwd.assert_not_called()
        ftp_client.cwd.assert_called_once_with("over there")

    @staticmethod
    def test_cwd_connection_not_open():
        """
        Test that CWD returns errors as expected if `cwd` is called when a connection
        is closed
        """
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.login = Mock(side_effect=ftp_client.__enter__)
        ftp_client.close = Mock(side_effect=ftp_client.__exit__)

        host = "what a great host"

        with FTPExtractor(dm, host) as ftp_session:
            ftp_session.cwd

        with pytest.raises(RuntimeError):
            ftp_session.cwd

        # TODO create a test for the cwd.setter

    @staticmethod
    def test_request(tmp_path):
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.retrbinary = Mock(side_effect=ftp_client.retrbinary)

        host = "what a great host"

        out_path = pathlib.PurePosixPath(tmp_path)
        with FTPExtractor(dm, host) as ftp:
            ftp.request(pathlib.PurePosixPath("two.dat"), out_path)

        assert ftp_client.host == host
        assert ftp_client.contexts == 0
        ftp_client.login.assert_called()
        assert ftp_client.commands == ["RETR two.dat"]

    @staticmethod
    def test_request_destination_is_not_a_directory(tmp_path):
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.retrbinary = Mock(side_effect=ftp_client.retrbinary)

        host = "what a great host"

        out_path = pathlib.PurePosixPath(tmp_path) / "himom.dat"
        with FTPExtractor(dm, host) as ftp:
            ftp.request(pathlib.PurePosixPath("two.dat"), out_path)

        assert ftp_client.host == host
        assert ftp_client.contexts == 0
        ftp_client.login.assert_called()
        assert ftp_client.commands == ["RETR two.dat"]

    @staticmethod
    def test_request_client_error(tmp_path):
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.retrbinary = Mock(side_effect=ftplib.error_perm)

        host = "what a great host"

        out_path = pathlib.PurePosixPath(tmp_path)
        with FTPExtractor(dm, host) as ftp:
            with pytest.raises(RuntimeError):
                ftp.request(pathlib.PurePosixPath("two.dat"), out_path)

        assert ftp_client.host == host
        assert ftp_client.contexts == 0
        ftp_client.login.assert_called()
        assert ftp_client.commands == []
