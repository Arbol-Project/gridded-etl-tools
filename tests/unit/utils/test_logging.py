import logging
import pathlib
import sys
from unittest.mock import Mock


class TestLogging:
    @staticmethod
    def test_log_to_file(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        mocker.patch("gridded_etl_tools.utils.logging.Logging.default_log_path")
        manager_class.default_log_path.return_value = "put/them/here.log"

        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = False
        formatter = mock_logging.Formatter.return_value

        handler = manager_class.log_to_file()

        assert handler is mock_logging.handlers.WatchedFileHandler.return_value
        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_called_once_with(logging.INFO)

        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with(
            "%(asctime)s <%(name)s in %(threadName)s> %(levelname)-8s %(message)s", "%Y/%m/%d %H:%M"
        )
        logger.addHandler.assert_called_once_with(handler)

        mock_logging.handlers.WatchedFileHandler.assert_called_once_with("put/them/here.log")

    @staticmethod
    def test_log_to_file_explicit_args(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        mocker.patch("gridded_etl_tools.utils.logging.Logging.default_log_path")
        manager_class.default_log_path.return_value = "put/them/here.log"

        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = False
        formatter = mock_logging.Formatter.return_value

        handler = manager_class.log_to_file("no/no/here.log", logging.CRITICAL, "like this", "and like this")

        assert handler is mock_logging.handlers.WatchedFileHandler.return_value
        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_called_once_with(logging.CRITICAL)

        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with("like this", "and like this")
        logger.addHandler.assert_called_once_with(handler)

        mock_logging.handlers.WatchedFileHandler.assert_called_once_with("no/no/here.log")

    @staticmethod
    def test_log_to_file_existing_handlers_but_no_match(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        mocker.patch("gridded_etl_tools.utils.logging.Logging.default_log_path")
        manager_class.default_log_path.return_value = "put/them/here.log"

        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = True
        formatter = mock_logging.Formatter.return_value

        handler = manager_class.log_to_file()

        assert handler is mock_logging.handlers.WatchedFileHandler.return_value
        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_called_once_with(logging.INFO)

        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with(
            "%(asctime)s <%(name)s in %(threadName)s> %(levelname)-8s %(message)s", "%Y/%m/%d %H:%M"
        )
        logger.addHandler.assert_called_once_with(handler)

        mock_logging.handlers.WatchedFileHandler.assert_called_once_with("put/them/here.log")

    @staticmethod
    def test_log_to_file_an_existing_handler_matches(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        mock_logging.handlers.WatchedFileHandler = logging.handlers.WatchedFileHandler
        mocker.patch("gridded_etl_tools.utils.logging.Logging.default_log_path")
        mocker.patch("gridded_etl_tools.utils.logging.Logging.info")
        manager_class.default_log_path.return_value = "put/them/here.log"

        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = True
        formatter = mock_logging.Formatter.return_value
        handler = Mock(spec=logging.handlers.WatchedFileHandler, baseFilename="put/them/here.log", level=logging.INFO)
        logger.handlers = ["not it", handler, "also not it"]

        assert manager_class.log_to_file() is handler

        handler.setFormatter.assert_called_once_with(formatter)
        mock_logging.getLogger.assert_called_once_with()
        logger.addHandler.assert_not_called()

    @staticmethod
    def test_log_to_console(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = False
        formatter = mock_logging.Formatter.return_value

        handler = manager_class.log_to_console()

        assert handler is mock_logging.StreamHandler.return_value
        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_called_once_with(logging.INFO)
        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with("%(levelname)-8s <%(name)s in %(threadName)s> %(message)s")
        logger.addHandler.assert_called_once_with(handler)
        mock_logging.StreamHandler.assert_called_once_with()

    @staticmethod
    def test_log_to_console_non_default_args(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = False
        formatter = mock_logging.Formatter.return_value

        handler = manager_class.log_to_console(logging.CRITICAL, "like this")

        assert handler is mock_logging.StreamHandler.return_value
        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_called_once_with(logging.CRITICAL)
        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with("like this")
        logger.addHandler.assert_called_once_with(handler)
        mock_logging.StreamHandler.assert_called_once_with()

    @staticmethod
    def test_log_to_console_existing_handlers_but_none_match(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = True
        logger.handlers = ["nope", "not it"]
        formatter = mock_logging.Formatter.return_value

        handler = manager_class.log_to_console()

        assert handler is mock_logging.StreamHandler.return_value
        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_called_once_with(logging.INFO)
        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with("%(levelname)-8s <%(name)s in %(threadName)s> %(message)s")
        logger.addHandler.assert_called_once_with(handler)
        mock_logging.StreamHandler.assert_called_once_with()

    @staticmethod
    def test_log_to_console_an_existing_handler_matches(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        mocker.patch("gridded_etl_tools.utils.logging.Logging.info")

        logger = mock_logging.getLogger.return_value
        logger.hasHandlers.return_value = True
        handler = Mock(
            stream=sys.stderr,
            level=logging.INFO,
        )
        logger.handlers = ["nope", handler, "not it"]
        formatter = mock_logging.Formatter.return_value

        assert manager_class.log_to_console() is handler

        handler.setFormatter.assert_called_once_with(formatter)
        handler.setLevel.assert_not_called()
        mock_logging.getLogger.assert_called_once_with()
        mock_logging.Formatter.assert_called_once_with("%(levelname)-8s <%(name)s in %(threadName)s> %(message)s")
        logger.addHandler.assert_not_called()
        mock_logging.StreamHandler.assert_not_called()

    @staticmethod
    def test_default_log_path(mocker, manager_class):
        mocker.patch("pathlib.Path.mkdir")
        assert manager_class.default_log_path(logging.INFO) == pathlib.Path("logs/DummyManager_INFO.log")
        pathlib.Path.mkdir.assert_called_once_with(pathlib.Path("logs"), mode=511, parents=False, exist_ok=True)

    @staticmethod
    def test_log(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        manager_class.log("Hi mom!", logging.CRITICAL, foo="bar")
        mock_logging.getLogger.assert_called_once_with("DummyManager")
        mock_logging.getLogger.return_value.log.assert_called_once_with(logging.CRITICAL, "Hi mom!", foo="bar")

    @staticmethod
    def test_info(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        manager_class.info("Hi mom!", foo="bar")
        mock_logging.getLogger.assert_called_once_with("DummyManager")
        mock_logging.getLogger.return_value.log.assert_called_once_with(mock_logging.INFO, "Hi mom!", foo="bar")

    @staticmethod
    def test_debug(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        manager_class.debug("Hi mom!", foo="bar")
        mock_logging.getLogger.assert_called_once_with("DummyManager")
        mock_logging.getLogger.return_value.log.assert_called_once_with(mock_logging.DEBUG, "Hi mom!", foo="bar")

    @staticmethod
    def test_warn(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        manager_class.warn("Hi mom!", foo="bar")
        mock_logging.getLogger.assert_called_once_with("DummyManager")
        mock_logging.getLogger.return_value.log.assert_called_once_with(mock_logging.WARN, "Hi mom!", foo="bar")

    @staticmethod
    def test_error(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        manager_class.error("Hi mom!", foo="bar")
        mock_logging.getLogger.assert_called_once_with("DummyManager")
        mock_logging.getLogger.return_value.log.assert_called_once_with(mock_logging.ERROR, "Hi mom!", foo="bar")

    @staticmethod
    def test_log_except_hook(mocker, manager_class):
        mock_logging = mocker.patch("gridded_etl_tools.utils.logging.logging")
        manager_class.log_except_hook("foo", "bar", "baz")
        mock_logging.getLogger.assert_called_once_with("DummyManager")
        mock_logging.getLogger.return_value.log.assert_called_once_with(
            mock_logging.ERROR, "Uncaught exception", exc_info=("foo", "bar", "baz")
        )
