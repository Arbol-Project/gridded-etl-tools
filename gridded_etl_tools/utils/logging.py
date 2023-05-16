import sys
import logging
import pathlib

from .attributes import Attributes


class Logging(Attributes):
    """
    A base class holding logging methods for Zarr ETLs
    """

    @classmethod
    def log_to_file(
        cls,
        path: str = None,
        level: str = logging.INFO,
        log_format: str = "%(asctime)s %(levelname)-8s <%(name)s> %(message)s",
        time_format: str = "%Y/%m/%d %H:%M",
    ) -> logging.Handler:
        """
        Attach a logging.handlers.WatchedFileHandler that will write log statements at `level` or higher to `path` in the requested formats.
        Since this attaches the constructed handler to the root logger, this is basically a convenience function for creating a standard logging
        module log handler and attaching it in the standard way. Since it is attached to the root logger, it will log all log statements sent to
        Python's logging module, including both this application's and its underlying module's. If called with all defaults, this will create a log
        in the current directory with the name based on this manager's name and the log level.

        If there is already a handler attached to the root logger writing the same level log statements to the same path, set its formatting to
        the requested format and do not attach a new handler.

        Parameters
        ----------
        path : str
            Path to a regular file to write log statements too, or None to build a default name in the current directory.
        level : str, optional
            Statements this level or higher will be printed (see logging module for levels). Defaults to 'logging.INFO'
        log_format : str, optional.
            The format of the log statement (see logging module for formatting options).
            Defaults to "%(asctime)s %(levelname)-8s <%(name)s> %(message)s"
        time_format:  str
            The format of the timestamp printed at the beginning of the statement. Defaults to "%Y/%m/%d %H:%M".

        Returns
        -------
        str
            The logging.Handler that will be handling writing to the log file (this can be useful for manually removing the
            handler from the root logger at a later time).

        """
        # If no path was specified, build a path using the manager name and log level in the current directory
        if path is None:
            path = cls.default_log_path(level)

        formatter = logging.Formatter(log_format, time_format)

        # Search for an existing handler that already satisfies the requirements of the passed arguments so repeat log statements don't get
        # logged. If an existing handler is found, set its formatting to the requested formatting and exit.
        handler_already_exists = False
        if logging.getLogger().hasHandlers():
            for handler in logging.getLogger().handlers:
                # Check if the handler is a WatchedFileHandler with the same level and path.
                if (
                    isinstance(handler, logging.handlers.WatchedFileHandler)
                    and handler.level == level
                    and pathlib.Path(handler.baseFilename).resolve()
                    == pathlib.Path(path).resolve()
                ):
                    handler_already_exists = True
                    handler.setFormatter(formatter)
                    cls.info(f"Found existing file log handler {handler}")
                    break

        if not handler_already_exists:
            # Use a standard file handler from the Python module
            handler = logging.handlers.WatchedFileHandler(path)

            # Apply requested formatting
            handler.setFormatter(formatter)

            # All log statments requested level or higher will be caught
            handler.setLevel(level)

            # Attach to root logger, catching every logger that doesn't have a handler attached to it, meaning this handler will catch
            # all log statements from this repository that go through `self.log` (assuming a handler hasn't been added manually somewhere)
            # and all log statements from Python modules that use standard logging practices (logging through the logging module and not
            # attaching a handler).
            logging.getLogger().addHandler(handler)

        return handler

    @classmethod
    def log_to_console(
        cls,
        level: str = logging.INFO,
        log_format: str = "%(levelname)-8s <%(name)s> %(message)s",
    ):
        """
        Attach a logging.StreamHandler that will write log statements at `level` or higher to the console. Since this attaches the stream handler
        to the root logger, this is basically a convenience function for creating a standard stream handler and attaching it in the standard way.
        Since it is attached to the root logger, it will log all log statements sent to Python's logging module, including both this application's
        and its underlying module's.

        If there is already a handler attached to the root logger writing to the console, set its formatting to the requested format and do not
        attach a new handler.

        Parameters
        ----------
        level : str, optional
            Statements of this level or higher will be printed (see logging module for levels). Defaults to `logging.INFO`.
        log_format : str, optional
            The format of the log statement (see logging module for formatting options). Defaults to "%(levelname)-8s <%(name)s> %(message)s"

        Returns
        -------
        str
            The `logging.Handler` that will write the log statements to console (this can be useful for removing the handler
            from the root logger manually at a later time).

        """
        formatter = logging.Formatter(log_format)

        # Search for an existing handler already writing to the console so repeat log statements don't get logged. If an existing handler is found,
        # set its formatting to the requested formatting and exit.
        handler_already_exists = False
        if logging.getLogger().hasHandlers():
            for handler in logging.getLogger().handlers:
                # Check if the attached handler has a stream equal to stderr or stdout, meaning it is writing to the console.
                if (
                    hasattr(handler, "stream")
                    and handler.level >= level
                    and handler.stream in (sys.stderr, sys.stdout)
                ):
                    handler_already_exists = True
                    # Apply requested formatting
                    handler.setFormatter(formatter)
                    cls.info(f"Found existing console log handler {handler}")
                    break

        if not handler_already_exists:
            # Uses sys.stderr as the stream to write to by default. It would also make sense to write to sys.stdout, but since either stream
            # could make sense, use the Python logging module default.
            handler = logging.StreamHandler()

            # Apply requested formatting
            handler.setFormatter(formatter)

            # All log statements requested level or higher will be caught.
            handler.setLevel(level)

            # See comment in self.log_to_file for information about what gets caught by this handler.
            logging.getLogger().addHandler(handler)

        return handler

    @classmethod
    def default_log_path(cls, level: str):
        """
        Returns a default log path in a "logs" directory within the current working directory, incorporating level into the name.
        Creates the "logs" directory if it doesn't already exist.
        Probably only useful internally.

        Parameters
        ----------
        level : str
            The logging level to use. See logging module for log levels.

        """
        pathlib.Path.mkdir(
            pathlib.Path("./logs"), mode=0o777, parents=False, exist_ok=True
        )
        return pathlib.Path("logs") / f"{cls.name()}_{logging.getLevelName(level)}.log"

    @classmethod
    def log(cls, message: str, level: str = logging.INFO, **kwargs):
        """
        This is basically a convenience function for calling logging.getLogger.log(`level`, `message`). The only difference is this uses
        the manager name as the name of the log to enable log statements that include the name of the manager instead of "root". Since
        logging module handlers at the root catch statements from all loggers that don't have a handler attached to them, the handlers
        attached to the root (either through logging.Logger.addHandler, self.log_to_file, or self.log_to_console) will catch the `message`
        logged with this function.

        Parameters
        ----------
        message : str
            Text to write to log.
        level : str
            Level of urgency of the message (see logging module for levels). Defaults to logging.INFO.
        **kwargs : dict
            Keyword arguments to forward to the logging.Logger. For example, `exc_info=True` can be used to print exception
            information. See the logging module for all keyword arguments.

        """
        logging.getLogger(cls.name()).log(level, message, **kwargs)

    @classmethod
    def info(cls, message: str, **kwargs):
        """
        Log a message at `logging.INFO` level.

        Parameters
        ----------
        message : str
            Text to write to log.
        **kwargs : dict
            Keywords arguments passed to `logging.Logger.log`.

        """
        cls.log(message, logging.INFO, **kwargs)

    @classmethod
    def debug(cls, message: str, **kwargs):
        """
        Log a message at `loggging.DEBUG` level.

        Parameters
        ----------
        message : str
            Text to write to log.
        **kwargs : dict
            Keywords arguments passed to `logging.Logger.log`.

        """
        cls.log(message, logging.DEBUG, **kwargs)

    @classmethod
    def warn(cls, message: str, **kwargs):
        """
        Log a message at `loggging.WARN` level.

        Parameters
        ----------
        message : str
            Text to write to log.
        **kwargs : dict
            Keywords arguments passed to `logging.Logger.log`.
        """
        cls.log(message, logging.WARN, **kwargs)

    @classmethod
    def error(cls, message: str, **kwargs):
        """
        Log a message at `logging.ERROR` level.

        Parameters
        ----------
        message : str
            Text to write to log.
        **kwargs : dict
            Keywords arguments passed to `logging.Logger.log`.

        """
        cls.log(message, logging.ERROR, **kwargs)

    @classmethod
    def log_except_hook(cls, exc_type, exc_value, exc_traceback):
        cls.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
