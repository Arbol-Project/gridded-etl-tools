import pytest
import time
import ftplib
import pathlib
import responses
import requests
import re
import os


from unittest.mock import Mock
from gridded_etl_tools.utils.extractor import (
    Extractor,
    HTTPExtractor,
    S3Extractor,
    S3ExtractorKerchunk,
    S3ExtractorDownload,
    S3ExtractorBase,
    FTPExtractor,
)
from .test_convenience import DummyFtpClient


class ConcreteExtractor(Extractor):
    """
    Make base class instantiable
    """

    def request(*args, **kwargs):
        pass

    def _get_local_file_from_remote(self, remote_file_path):
        pass

    def _get_local_file_path(self, remote_file_path, local_path):
        pass


class TestExtractor:
    @staticmethod
    def test_init():
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        assert extractor.dm == dm

    def test_pool_no_jobs(self):
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        extractor.request = Mock(return_value=True)
        result = extractor.pool(batch=[])
        assert extractor.request.call_count == 0
        assert result is False

    def test_pool_request_success(self):
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        extractor.request = Mock(return_value=True)

        # Mappings
        result = extractor.pool(batch=[{"one": 1, "two": 2}, {"one": 3, "two": 4}, {"one": 5, "two": 6}])
        assert extractor.request.call_count == 3
        extractor.request.assert_any_call(one=1, two=2)
        extractor.request.assert_any_call(one=3, two=4)
        extractor.request.assert_any_call(one=5, two=6)

        # Sequences
        result = extractor.pool(batch=[[1, 2], [3, 4], [5, 6]])
        assert extractor.request.call_count == 6
        extractor.request.assert_any_call(1, 2)
        extractor.request.assert_any_call(3, 4)
        extractor.request.assert_any_call(5, 6)

        # Scalars
        result = extractor.pool(batch=[1, 2, 3, 4])
        assert extractor.request.call_count == 10
        extractor.request.assert_any_call(1)
        extractor.request.assert_any_call(2)
        extractor.request.assert_any_call(3)
        extractor.request.assert_any_call(4)

        assert result is True

    def test_pool_request_accept_any_success(self, caplog):
        dm = Mock()
        extractor = ConcreteExtractor(dm)

        def mixed_request_success(*args, **kwargs):
            return [True, False, True]

        extractor.request = Mock(side_effect=mixed_request_success)
        result = extractor.pool(
            batch=[{"one": 1, "two": 2}, {"one": 3, "two": 4}, {"one": 5, "two": 6}], accept_any_success=True
        )
        assert extractor.request.call_count == 3
        assert result is True
        assert "Some requests succeeded, some did not. Check the logs for details of which." in caplog.text

    def test_pool_request_failure(self):
        dm = Mock()
        extractor = ConcreteExtractor(dm)
        extractor.request = Mock(return_value=False)
        result = extractor.pool(batch=[{"one": 1, "two": 2}, {"one": 3, "two": 4}, {"one": 5, "two": 6}])
        assert extractor.request.call_count == 3
        assert result is False


# Mock data for testing HTTP extraction

government_website = "https://bizarre-foods.reference/farmland/yield.html"

government_website_broken = "https://bizarre-foods.reference/broken/link"

crops = b"""<a class="example" href="purple-coffee" id="Link1">Purple coffee<</a>,
    <a class="example" href="sour-avocadoes" id="Link2"><Sour avocadoes</a>,
    <a class="example" href="happy/juice.data" id="Link2"><Happy Juice</a>,
    <a class="example" href="/deforestation_techniques.html" id="Link2">Learn from the pros</a>,
    <a class="example" href="https://boring-normal-foods.tasty" id="Link1"><Other stuff</a>,
    <a class="example" href="mailto:vladimir-putin@farmers-only.com" id="Link1"><Lonely hearts</a>,"""

happy_juice = b"\xe2\x98\xba\xf0\x9f\xa7\x83"
purple_coffee = b"\xf0\x9f\x9f\xaa\xe2\x98\x95"
sour_avocadoes = b"\xf0\x9f\x8d\x8b\xf0\x9f\xa5\x91"


# Requests for mock URLs defined in this fixture will be successfully requested when this fixture is passed to a
# function and the `@responses.activate` header is used.
@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        # To test successful requests
        responses.get(government_website, body=crops, status=200)

        # To test unsuccessful requests
        responses.get(government_website_broken, status=500)

        # To test downloading data
        responses.get("https://bizarre-foods.reference/farmland/sour-avocadoes", body=sour_avocadoes, status=200)
        responses.get("https://bizarre-foods.reference/farmland/purple-coffee", body=purple_coffee, status=200)
        responses.get("https://bizarre-foods.reference/farmland/happy/juice.data", body=happy_juice, status=200)

        yield rsps


class TestHTTPExtractor:
    @staticmethod
    def test_member_assignments(manager_class):
        with HTTPExtractor(manager_class(), 4, 5, 0.5) as extractor:
            assert extractor._concurrency_limit == 4
            assert extractor.retries == 5
            assert extractor.backoff_factor == 0.5

    # Test a request that gets the mocked responses. This will write a file to a temporary location using pytest's
    # `tmp_path` fixture.
    @staticmethod
    @responses.activate
    def test_requests(manager_class, mocked_responses, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        with HTTPExtractor(manager_class(), 4, 3, 0.1) as extractor:
            # Test writing to a given file path
            extractor.request(government_website, tmp_path / "automated_insurance.data")
            assert (tmp_path / "automated_insurance.data").read_bytes() == crops

            # Change into a temp directory to test the default write location. The try/finally ensures that the original
            # directory is always returned to. In Python 3.11, there is a better alternative in contextlib.chdir.
            original_working_dir = os.getcwd()
            try:
                os.chdir(tmp_path)
                extractor.request(government_website)
            finally:
                os.chdir(original_working_dir)
            assert (tmp_path / "yield.html").read_bytes() == crops

            # Test writing to an existing directory
            research_folder = tmp_path / "research"
            research_folder.mkdir(exist_ok=True)
            extractor.request(government_website, research_folder)
            assert (research_folder / "yield.html").read_bytes() == crops

    # Test that a 500 error fails with a RetryError exception
    @staticmethod
    @responses.activate
    def test_server_error_request(manager_class, mocked_responses):
        with HTTPExtractor(manager_class(), 4, 3, 0.1) as extractor:
            with pytest.raises(requests.exceptions.RetryError):
                extractor.request(government_website_broken)

    # Test that making requests outside of a context manager fails
    @staticmethod
    def test_missing_context_manager(manager_class):
        extractor = HTTPExtractor(manager_class(), 4, 3, 0.1)
        with pytest.raises(RuntimeError):
            extractor.request(government_website)
        with pytest.raises(RuntimeError):
            extractor.get_links(government_website)

    # Test link scraping and filtering
    @staticmethod
    @responses.activate
    def test_get_links(manager_class, mocked_responses, tmp_path):
        with HTTPExtractor(manager_class(), 4, 3, 0.1) as extractor:
            # Get all links
            links = extractor.get_links(government_website)
            assert links == set(
                [
                    "https://bizarre-foods.reference/farmland/sour-avocadoes",
                    "https://bizarre-foods.reference/farmland/purple-coffee",
                    "https://bizarre-foods.reference/farmland/happy/juice.data",
                    "https://bizarre-foods.reference/deforestation_techniques.html",
                    "https://boring-normal-foods.tasty",
                    "mailto:vladimir-putin@farmers-only.com",
                ]
            )

            # Get links that match a regular expression
            links = extractor.get_links(government_website, re.compile("[a-z]+-[a-z]+-[a-z]+"))
            assert links == set(["https://boring-normal-foods.tasty"])

            # Remove links matching specific prefixes using a separately defined function
            def filter_out_prefixes(href):
                return not href.startswith("https://") and not href.startswith("mailto:") and not href.startswith("/")

            links = extractor.get_links(government_website, filter_out_prefixes)
            assert links == set(
                [
                    "https://bizarre-foods.reference/farmland/sour-avocadoes",
                    "https://bizarre-foods.reference/farmland/purple-coffee",
                    "https://bizarre-foods.reference/farmland/happy/juice.data",
                ]
            )

            # Download the content of the last request, using a temp directory as the working directory. The
            # try/finally ensures that the original directory is always returned to. In Python 3.11, there is a better
            # alternative in contextlib.chdir.
            tmp_path.mkdir(exist_ok=True)
            original_working_dir = os.getcwd()
            try:
                os.chdir(tmp_path)
                extractor.pool(links)
            finally:
                os.chdir(original_working_dir)

            # Check the files
            (tmp_path / "sour-avocadoes").read_bytes().decode() == "🍋🥑"
            (tmp_path / "purple-coffee").read_bytes().decode() == "🟪☕"
            (tmp_path / "juice.data").read_bytes().decode() == "☺🧃"

    # Use OrderedRegistry to create a series of only failed responses, and a series of failed responses followed
    # by a successful response
    @staticmethod
    @responses.activate(registry=responses.registries.OrderedRegistry)
    def test_extract_http_retries(manager_class, tmp_path):
        with HTTPExtractor(manager_class(), 4, 3, 0.1) as extractor:
            results = list()
            for index in range(0, 6):
                results.append(responses.get("https://good.data", status=500))
            with pytest.raises(requests.exceptions.RetryError):
                extractor.get_links("https://good.data")
            for index, result in enumerate(results):
                if index < 4:
                    assert result.call_count == 1
                else:
                    assert result.call_count == 0
        responses.reset()
        with HTTPExtractor(manager_class(), 4, 3, 0.1) as extractor:
            results = list()
            for index in range(0, 3):
                results.append(responses.get("https://good.data", status=500))
            results.append(responses.get("https://good.data", status=200))
            assert extractor.request("https://good.data", tmp_path / pathlib.Path("stonks.jpeg"))
            for result in results:
                assert result.call_count == 1


class TestS3ExtractorKerchunk:
    @staticmethod
    def test_s3_misspecified_extraction_errors(manager_class):
        dm = manager_class()

        # Test that ignorable extraction errors are converted to a tuple
        ignorable_extraction_errors = [FileNotFoundError]
        extractor = S3ExtractorKerchunk(dm, ignorable_extraction_errors=ignorable_extraction_errors)
        assert extractor.ignorable_extraction_errors == tuple(ignorable_extraction_errors)

        # Ensure the same for unsupported extraction errors
        unsupported_extraction_errors = [ValueError]
        extractor = S3ExtractorKerchunk(dm, unsupported_extraction_errors=unsupported_extraction_errors)
        assert extractor.unsupported_extraction_errors == tuple(unsupported_extraction_errors)

    @staticmethod
    def test_s3_request(manager_class):
        extractor = S3ExtractorKerchunk(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]
        kwargs = {"file_path": rfp, "scan_indices": 0, "local_file_path": lfp}

        extractor.dm.kerchunkify = Mock(autospec=True)

        extractor.request(*args)
        extractor.dm.kerchunkify.assert_called_once_with(**kwargs)

    @staticmethod
    def test_s3_request_with_informative_id(manager_class):
        extractor = S3ExtractorKerchunk(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, "informative information"]
        kwargs = {"file_path": rfp, "scan_indices": 0, "local_file_path": lfp}

        extractor.dm.kerchunkify = Mock(autospec=True)

        extractor.request(*args)
        extractor.dm.kerchunkify.assert_called_once_with(**kwargs)

    @staticmethod
    def test_s3_request_remote_file_is_not_on_s3(manager_class):
        extractor = S3ExtractorKerchunk(manager_class())

        rfp = "t4://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extractor.dm.kerchunkify = Mock(autospec=True)

        with pytest.raises(ValueError):
            extractor.request(*args)

        extractor.dm.kerchunkify.assert_not_called()

    @staticmethod
    def test_s3_request_fail(manager_class):
        extract = S3ExtractorKerchunk(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extract.dm.kerchunkify = Mock(side_effect=Exception("mocked error"))
        time.sleep = Mock()  # avoid actually sleeping for large period of time

        with pytest.raises(FileNotFoundError):
            extract.request(*args)
        assert time.sleep.call_count == 5

    @staticmethod
    def test_s3_request_permitted_fail(manager_class):
        extract = S3ExtractorKerchunk(
            manager_class(),
            ignorable_extraction_errors=[FileNotFoundError],
            unsupported_extraction_errors=[ValueError],
        )

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extract.dm.kerchunkify = Mock(side_effect=FileNotFoundError("this is error is OK"))
        time.sleep = Mock()  # avoid actually sleeping for large period of time

        assert not extract.request(*args)
        assert time.sleep.call_count == 0

    @staticmethod
    def test_s3_request_unpermitted_fail(manager_class):
        extract = S3ExtractorKerchunk(
            manager_class(),
            ignorable_extraction_errors=[FileNotFoundError],
            unsupported_extraction_errors=[ValueError],
        )

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = "/local/sand/depo/castle1.json"
        args = [rfp, 0, 5, lfp, None]

        extract.dm.kerchunkify = Mock(side_effect=ValueError("this error is not OK"))
        time.sleep = Mock()  # avoid actually sleeping for large period of time

        with pytest.raises(ValueError):
            extract.request(*args)
        assert time.sleep.call_count == 0


class TestS3ExtractorDeprecation:
    @staticmethod
    def test_s3_extractor_deprecation_warning(manager_class, caplog):
        """Test that S3Extractor raises a deprecation warning"""
        dm = manager_class()

        # Clear any existing log records
        caplog.clear()

        # Create S3Extractor instance (should trigger deprecation warning)
        extractor = S3Extractor(dm)

        # Verify the warning was logged
        assert "S3Extractor is deprecated" in caplog.text
        assert "S3ExtractorKerchunk for kerchunking operations" in caplog.text
        assert "S3ExtractorDownload for direct file downloads" in caplog.text

        # Verify it still behaves like S3ExtractorKerchunk
        assert isinstance(extractor, S3ExtractorKerchunk)
        assert hasattr(extractor.dm, "zarr_jsons")


class TestS3ExtractorDownload:
    @staticmethod
    def test_s3_download_with_fs_provided(manager_class, tmp_path):
        """Test successful S3 file download with provided fs"""

        mock_s3_fs = Mock()
        extractor = S3ExtractorDownload(manager_class(), fs=mock_s3_fs)
        assert extractor.s3_fs == mock_s3_fs

        # Test that extract_from_s3 uses the provided fs
        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = tmp_path / "castle1.grib"
        extractor.perform_extraction(rfp, local_file_path=lfp)
        mock_s3_fs.download.assert_called_once_with(rfp, str(lfp))

    @staticmethod
    def test_s3_download_without_fs_provided(manager_class, tmp_path, monkeypatch):
        mock_s3fs = Mock()
        monkeypatch.setattr("s3fs.S3FileSystem", mock_s3fs)

        extractor = S3ExtractorDownload(manager_class())
        mock_s3fs.assert_called_once_with()
        assert extractor.s3_fs == mock_s3fs.return_value

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = tmp_path / "castle1.grib"
        extractor.perform_extraction(rfp, local_file_path=lfp)
        extractor.s3_fs.download.assert_called_once_with(rfp, str(lfp))

        # fs=None should also instantiate a new S3FileSystem
        S3ExtractorDownload(manager_class(), fs=None)
        assert mock_s3fs.call_count == 2

    @staticmethod
    def test_s3_download_request_success(manager_class, tmp_path):
        """Test successful S3 file download"""

        extractor = S3ExtractorDownload(manager_class())
        mock_s3_fs = Mock()
        extractor.s3_fs = mock_s3_fs

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = tmp_path / "castle1.grib"

        result = extractor.request(rfp, local_file_path=lfp)

        assert result is True
        mock_s3_fs.download.assert_called_once_with(rfp, str(lfp))

    @staticmethod
    def test_s3_download_request_no_local_path(manager_class):
        """Test that S3ExtractorDownload requires local_file_path"""
        extractor = S3ExtractorDownload(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"

        with pytest.raises(ValueError, match="local_file_path is required"):
            extractor.request(rfp)

    @staticmethod
    def test_s3_download_request_invalid_s3_path(manager_class):
        """Test that non-S3 paths are rejected"""
        extractor = S3ExtractorDownload(manager_class())

        rfp = "t4://bucket/sand/castle/castle1.grib"
        lfp = pathlib.Path("/local/sand/depo/castle1.grib")

        with pytest.raises(ValueError, match="is not an S3 path"):
            extractor.request(rfp, local_file_path=lfp)

    @staticmethod
    def test_s3_download_scan_indices_invalid(manager_class, monkeypatch, tmp_path):
        """Test that scan_indices parameter raises an exception for downloads"""
        # Mock s3fs module
        mock_s3fs_module = Mock()
        mock_filesystem = Mock()
        mock_s3fs_module.S3FileSystem.return_value = mock_filesystem

        # Mock the s3fs import
        monkeypatch.setattr("s3fs.S3FileSystem", mock_s3fs_module.S3FileSystem)

        extractor = S3ExtractorDownload(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = tmp_path / "castle1.grib"

        with pytest.raises(ValueError, match="scan_indices not supported"):
            extractor.request(rfp, scan_indices=(1, 2), local_file_path=lfp)

    @staticmethod
    def test_s3_download_retry_logic(manager_class, monkeypatch, tmp_path):
        """Test that retry logic works for download failures"""
        # Mock s3fs to fail then succeed
        mock_s3fs_module = Mock()
        mock_filesystem = Mock()
        mock_filesystem.download.side_effect = [Exception("Network error"), None]
        mock_s3fs_module.S3FileSystem.return_value = mock_filesystem

        # Mock the s3fs import and time.sleep
        monkeypatch.setattr("s3fs.S3FileSystem", mock_s3fs_module.S3FileSystem)
        monkeypatch.setattr("time.sleep", Mock())

        extractor = S3ExtractorDownload(manager_class())

        rfp = "s3://bucket/sand/castle/castle1.grib"
        lfp = tmp_path / "castle1.grib"

        result = extractor.request(rfp, tries=2, local_file_path=lfp)

        assert result is True
        assert mock_filesystem.download.call_count == 2


class TestS3ExtractorBase:
    @staticmethod
    def test_s3_extractor_base_is_abstract():
        """Test that S3ExtractorBase cannot be instantiated directly"""
        dm = Mock()

        with pytest.raises(TypeError):
            S3ExtractorBase(dm)


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

        out_path = pathlib.Path(tmp_path)
        with FTPExtractor(dm, host) as ftp:
            ftp.request(pathlib.PurePosixPath("two.dat"), out_path)

        assert ftp_client.host == host
        assert ftp_client.contexts == 0
        ftp_client.login.assert_called()
        assert ftp_client.commands == ["RETR two.dat"]

    @staticmethod
    def test_request_local_is_not_a_directory(tmp_path):
        dm = Mock()
        ftp_client = ftplib.FTP = DummyFtpClient()
        ftp_client.retrbinary = Mock(side_effect=ftp_client.retrbinary)

        host = "what a great host"

        out_path = pathlib.Path(tmp_path) / "himom.dat"
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

        out_path = pathlib.Path(tmp_path)
        with FTPExtractor(dm, host) as ftp:
            with pytest.raises(RuntimeError):
                ftp.request(pathlib.PurePosixPath("two.dat"), out_path)

        assert ftp_client.host == host
        assert ftp_client.contexts == 0
        ftp_client.login.assert_called()
        assert ftp_client.commands == []
