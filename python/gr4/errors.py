"""The exceptions gr4 raises.

Every failure the runtime reports is a gr4.Error, the extension's own exception class, with the runtime's message,
the place it names (where) and the time it gives (time_ns). Most operations raise the subclass below that names the
kind of failure, and each subclass also derives from the built-in exception for that kind. Graph.remove(),
Block.interior, the description of a type whose default construction fails and a request's error reply raise Error
itself.
"""

import contextlib
from collections.abc import Iterator

from . import _gr4

Error = _gr4.Error


class UnknownBlockType(Error, LookupError):
    """A registry key that names no block or scheduler type this process can build."""


class PortError(Error, LookupError):
    """A port that does not exist, points the wrong way, or cannot be connected, disconnected or exported."""


class SettingError(Error, ValueError):
    """A setting a block refused."""


class GraphFileError(Error, ValueError):
    """A graph document that does not load; a parse failure names the line and column."""


class GraphBusyError(Error):
    """A structural edit of a graph while its scheduler runs it."""


class SchedulerError(Error):
    """A scheduler that cannot be created, cannot change state, or is closed."""


class RunError(Error):
    """A run that ended in error.

    events holds the error events the graph reported during the run, oldest first.
    """

    events: tuple = ()


class Timeout(Error, TimeoutError):
    """A request whose reply did not arrive within its timeout."""


@contextlib.contextmanager
def translated(kind: type[Error]) -> Iterator[None]:
    """Raises an Error of the extension inside the block as kind, with the same message, where and time_ns.

    An exception that already is a subclass of Error, and any other exception, passes unchanged.
    """
    try:
        yield
    except Error as error:
        if type(error) is not Error:
            raise
        raise kind(error.message, error.where, error.time_ns).with_traceback(error.__traceback__) from None
