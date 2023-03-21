
Logging
-------

Managers have built-in logging capabilities that are a wrapper around Python's `logging` module. By default, log statements will just be printed to the console, but log files can be enabled using the member function `DatasetManager.log_to_file`. Log statements from underlying modules will be logged as well.

    # Initialize c and start logging to the console (note, console_log is True by default anyway)
    c = MyClimateSet(console_log=True)

    # Add an INFO log (the default level is INFO)
    c.log_to_file()
    
    # Add a DEBUG log
    c.log_to_file(level=logging.DEBUG)

    # Prints "Hello, World!" to console and writes it to both ./[c.name()]_info.log and ./[c.name()]_debug.log
    c.debug("Hello, World!")

The [Python logging module](https://docs.python.org/3/library/logging.html) can be used directly to get more control over logging. Any handlers or filters added to the [logging module's root log](https://docs.python.org/3/howto/logging.html#:~:text=The%20root%20of%20the%20hierarchy,methods%20have%20the%20same%20signatures.) will catch log statements automatically.
