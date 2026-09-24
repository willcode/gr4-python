"""Schedulers: running a graph, stopping it, and the events, requests and subscriptions of its message plane.

A Scheduler takes a Graph and runs it on a thread the extension owns. Every call that waits (run, wait, events,
request) drains what the graph reported: a reply goes to the request that waits for it, an error event to errors and to
the queue events() reads, and every other event to that queue, which holds EVENT_QUEUE_LENGTH events and drops the
oldest. A reply that arrives after its request timed out is dropped. The notifications of a subscription with a
callback go to its callback, which runs on the scheduler's dispatch thread; that thread exists while a callback
subscription does and drains every DISPATCH_INTERVAL seconds.

An atexit hook closes every live scheduler before the interpreter and the plugin loader are torn down, since a
scheduler's destructor waits for its workers.
"""

import atexit
import collections
import dataclasses
import itertools
import queue
import sys
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import TracebackType

from . import _gr4, registry
from .errors import Error, RunError, SchedulerError, Timeout, translated
from .graph import Block, Graph
from .values import Settings, coercion_types, unknown_setting

__all__ = ["Command", "Event", "Scheduler", "State", "Subscription"]

State = _gr4.State
Command = _gr4.Command

EVENT_QUEUE_LENGTH = 4096
"""The most events the queue events() reads holds; a drain past it drops the oldest."""

WAIT_SLICE = 0.1
"""The longest a waiting call waits between two drains, in seconds; KeyboardInterrupt arrives between two slices."""

DISPATCH_INTERVAL = 0.02
"""The longest the dispatch thread waits between two drains, in seconds."""

CLOSE_TIMEOUT = 1.0
"""The longest close() waits for a running callback to return, in seconds; the exit hook closes through close()."""

EXPIRED_REQUESTS = 1024
"""The most ids of timed-out requests a scheduler keeps; a late reply to one of them is dropped."""

SETTINGS_ENDPOINTS = ("Settings", "StagedSettings")
"""The endpoints of a block whose payload request() and send() convert to the block's setting types."""

_ids = itertools.count(1)


def _fresh_id(kind: str) -> str:
    """Returns a request id no other request or subscription of this process carries."""
    return f"gr4-{kind}-{next(_ids)}"


@dataclasses.dataclass(frozen=True)
class Event:
    """One report of the running graph.

    source is the unique name of the reporting block and endpoint the property it reported on. command is FINAL for
    the reply to a request and NOTIFY for a notification, and request_id is the id of the request or subscription the
    event answers, or ''. An error event carries is_error, text and the time it was raised in time_ns; any other
    event carries data.
    """

    source: str
    endpoint: str
    command: Command
    request_id: str
    is_error: bool
    text: str
    data: dict
    time_ns: int

    @classmethod
    def from_runtime(cls, event: _gr4.RuntimeEvent) -> "Event":
        """Returns the Event a RuntimeEvent of the extension describes."""
        return cls(
            event.source,
            event.endpoint,
            Command(event.command),
            event.client_request_id,
            event.is_error,
            event.text,
            {} if event.is_error else event.data,
            event.time_ns,
        )

    def __str__(self) -> str:
        """The source, endpoint and command, then the error text or the data."""
        body = f"error: {self.text}" if self.is_error else repr(self.data)
        return f"{self.source} {self.endpoint} {self.command.name} {body}"


class _Reply:
    """The slot a drain fills with the reply to one pending request."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.event: Event | None = None


class Subscription:
    """A subscription to the notifications a block makes on one endpoint.

    With a callback, each notification is passed to the callback on the scheduler's dispatch thread; without one,
    notifications go to the queue events() reads. A context manager that cancels on exit.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        service: str,
        endpoint: str,
        callback: Callable[[Event], object] | None,
        request_id: str,
    ) -> None:
        """Records a subscription; Scheduler.subscribe() and Block.subscribe() create them."""
        self.scheduler = scheduler
        self.service = service
        self.endpoint = endpoint
        self.callback = callback
        self.request_id = request_id
        self._active = True

    @property
    def active(self) -> bool:
        """True until the subscription is canceled or its scheduler closed."""
        return self._active

    def cancel(self) -> None:
        """Ends the subscription. A second call does nothing."""
        self.scheduler._unsubscribe(self)

    def __enter__(self) -> "Subscription":
        """Returns the subscription."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        """Cancels the subscription and lets an exception propagate."""
        self.cancel()
        return False

    def __repr__(self) -> str:
        """Names the block, the endpoint and the id."""
        return f"<Subscription {self.service} {self.endpoint} {self.request_id}{'' if self._active else ' canceled'}>"


_live: "weakref.WeakSet[Scheduler]" = weakref.WeakSet()


def _service_name(service: "str | Block | Scheduler") -> str:
    if isinstance(service, (Block, Scheduler)):
        return service.unique_name
    if isinstance(service, str):
        return service
    raise TypeError(f"a service is a Block, a Scheduler or a unique name, got {service!r}")


def _payload_types(service: "str | Block | Scheduler", endpoint: str, payload: Mapping | None) -> dict[str, str] | None:
    """Returns the type codes of a payload for the settings endpoints of a Block, and None for any other message."""
    if not isinstance(service, Block) or endpoint not in SETTINGS_ENDPOINTS or not payload:
        return None
    return coercion_types(service._handle.setting_types(), payload)


class _SchedulerSettings(Settings):
    """The settings of a scheduler, read like a block's. A scheduler applies its settings when it is created."""

    def update(self, values: Mapping[str, object] | Iterable[tuple[str, object]] = (), /, **settings: object) -> None:
        """Raises SchedulerError naming the construction settings, where a scheduler's settings are given."""
        raise SchedulerError(
            f"{self._owner} applies its settings when it is created; give them to gr4.Scheduler(graph, type, "
            "**settings)",
            "Scheduler.settings",
        )


class Scheduler:
    """A scheduler of the registry holding a graph.

    run() runs the graph to its end. start() starts it and returns the scheduler, a context manager that stops the
    graph on exit: ``with scheduler.start():`` runs a block of code beside the graph. A stopped scheduler runs its
    graph again.
    """

    def __init__(self, graph: Graph, type: str | None = None, **settings: object) -> None:
        """Instantiates the scheduler registered as type, the default scheduler when None, and takes the graph.

        The settings are the scheduler's own, checked against its description as Graph.add() checks a block's. The
        graph moves into the scheduler, and the Graph object then views the scheduler's graph.
        """
        if not isinstance(graph, Graph):
            raise TypeError(f"a Scheduler takes a Graph, got {graph!r}")
        if graph._root is not graph:
            raise SchedulerError("a scheduler takes a whole graph, not the interior of a block", "Scheduler")
        if graph._scheduler is not None:
            raise SchedulerError("the graph belongs to a scheduler already", "Scheduler")
        key = registry.DEFAULT_SCHEDULER if type is None else type
        declared = {setting.name: setting.type for setting in registry[key].settings.values()}
        for name in settings:
            if name not in declared:
                raise unknown_setting(name, declared, key)
        with translated(SchedulerError):
            self._runtime = _gr4.Runtime.create(graph._graph, key, settings, coercion_types(declared, settings))
        self.type = key
        self.graph = graph
        graph._graph = self._runtime.graph()
        graph._scheduler = self
        self._lock = threading.RLock()
        self._closed = False
        self._pending: dict[str, _Reply] = {}
        self._expired: dict[str, None] = {}
        self._errors: list[Event] = []
        self._queue: collections.deque[Event] = collections.deque(maxlen=EVENT_QUEUE_LENGTH)
        self._subscriptions: dict[str, Subscription] = {}
        self._deliveries: queue.SimpleQueue[tuple[Subscription, Event]] = queue.SimpleQueue()
        self._dispatcher: threading.Thread | None = None
        self._dispatch_stop = threading.Event()
        _live.add(self)

    def _check_open(self) -> None:
        if self._closed:
            raise SchedulerError("the scheduler is closed", "Scheduler")

    def _in_run(self) -> bool:
        """True while a run started by run() or start() has not ended."""
        return not self._closed and not self._runtime.wait(0.0)

    @property
    def unique_name(self) -> str:
        """The scheduler's unique name, the service a request to its own endpoints addresses."""
        self._check_open()
        return self._runtime.scheduler().unique_name

    @property
    def state(self) -> State:
        """The scheduler's lifecycle state."""
        self._check_open()
        return State(self._runtime.state)

    @property
    def settings(self) -> Settings:
        """The scheduler's own settings. A scheduler applies its settings when it is created; staging raises
        SchedulerError.
        """
        self._check_open()
        return _SchedulerSettings(self._runtime.scheduler(), f"scheduler '{self.type}'")

    @property
    def errors(self) -> list[Event]:
        """Every error event of the current or last run, oldest first."""
        self._drain()
        with self._lock:
            return list(self._errors)

    def _begin(self) -> None:
        self._check_open()
        with self._lock:
            self._errors.clear()
        with translated(SchedulerError):
            self._runtime.run_in_thread()

    def _finish(self) -> None:
        """Raises RunError when the last run ended in error: its message and where name the first error event."""
        self._drain()
        try:
            self._runtime.result()
        except Error as error:
            events = tuple(self.errors)
            if events:
                first = events[0]
                failure = RunError(f"{first.source}: {first.text}; {error.message}", first.source, first.time_ns)
            else:
                failure = RunError(error.message, error.where, error.time_ns)
            failure.events = events
            raise failure.with_traceback(error.__traceback__) from None

    def _halt(self) -> None:
        self._runtime.stop()
        self._drain()

    def run(self) -> None:
        """Runs the graph to its end and returns.

        RunError when the run ended in error; it names the first error event and carries the error events of the
        run. On KeyboardInterrupt the graph is stopped and its run waited for before the interrupt propagates.
        """
        try:
            self._begin()
            while not self._runtime.wait(WAIT_SLICE):
                self._drain()
        except KeyboardInterrupt:
            self._halt()
            raise
        self._finish()

    def start(self) -> "Scheduler":
        """Starts the graph on the extension's thread and returns the scheduler, a context manager that stops it.

        The context manager raises RunError on exit when the run ended in error.
        """
        self._begin()
        return self

    def stop(self) -> None:
        """Stops the graph and returns once its run has ended. Without a run in progress, and on a closed scheduler,
        it does nothing.
        """
        if not self._closed:
            self._halt()

    def pause(self) -> None:
        """Requests a pause of the running graph; state reaches PAUSED."""
        self._check_open()
        with translated(SchedulerError):
            self._runtime.request_state(State.REQUESTED_PAUSE)

    def resume(self) -> None:
        """Resumes a paused graph; state returns to RUNNING."""
        self._check_open()
        with translated(SchedulerError):
            self._runtime.request_state(State.RUNNING)

    def wait(self, timeout: float | None = None) -> bool:
        """Waits for the run to end, without limit when timeout is None.

        Returns True when the run has ended, or when there is none, and False when timeout seconds passed first.
        RunError when the run ended in error.
        """
        self._check_open()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            slice_ = WAIT_SLICE if deadline is None else max(0.0, min(WAIT_SLICE, deadline - time.monotonic()))
            if self._runtime.wait(slice_):
                break
            self._drain()
            if deadline is not None and time.monotonic() >= deadline:
                return False
        self._finish()
        return True

    def __enter__(self) -> "Scheduler":
        """Returns the scheduler."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        """Stops the graph. RunError when the run ended in error and the with block raised nothing; an exception
        raised in the with block propagates.
        """
        self.stop()
        if exc_type is None and not self._closed:
            self._finish()
        return False

    def _drain(self) -> None:
        """Routes what the runtime reported. An exception raised while one event is routed, KeyboardInterrupt among
        them, propagates once the rest of its batch is routed.
        """
        with self._lock:
            if self._closed:
                return
            while batch := self._runtime.poll_events(1024):
                events = iter(batch)
                try:
                    for raw in events:
                        self._route(Event.from_runtime(raw))
                finally:
                    for raw in events:
                        self._route(Event.from_runtime(raw))

    def _route(self, event: Event) -> None:
        if event.request_id:
            reply = self._pending.get(event.request_id)
            if reply is not None and reply.event is None:
                reply.event = event
                reply.done.set()
                return
            if event.request_id in self._expired:
                del self._expired[event.request_id]
                return
        if event.is_error:
            self._errors.append(event)
            self._queue.append(event)
            return
        subscription = self._subscriptions.get(event.request_id) if event.request_id else None
        if subscription is not None and subscription.callback is not None:
            self._deliveries.put((subscription, event))
            return
        self._queue.append(event)

    def events(self, timeout: float | None = None) -> Iterator[Event]:
        """Yields what the graph reported since the last read, oldest first.

        With timeout None the iteration ends when the queue is empty. Otherwise it keeps yielding events as they arrive
        until timeout seconds have passed or the run has ended.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._drain()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    event = self._queue.popleft()
                yield event
            if deadline is None or self._closed:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._runtime.wait(min(DISPATCH_INTERVAL, remaining)):
                self._drain()
                with self._lock:
                    rest = list(self._queue)
                    self._queue.clear()
                yield from rest
                return

    def send(
        self,
        service: "str | Block | Scheduler",
        endpoint: str,
        payload: Mapping[str, object] | None = None,
        command: Command = Command.SET,
    ) -> None:
        """Sends one message to a block, a unique name or the scheduler itself, and returns without a reply.

        The payload converts as request() converts it.
        """
        self._send(command, _service_name(service), endpoint, payload, "", _payload_types(service, endpoint, payload))

    def _send(
        self,
        command: Command,
        service: str,
        endpoint: str,
        payload: Mapping | None,
        request_id: str,
        types: dict[str, str] | None = None,
    ) -> None:
        self._check_open()
        with translated(SchedulerError):
            self._runtime.send(
                command, service, endpoint, None if payload is None else dict(payload), request_id, types
            )

    def request(
        self,
        service: "str | Block | Scheduler",
        endpoint: str,
        payload: Mapping[str, object] | None = None,
        command: Command = Command.GET,
        timeout: float = 1.0,
    ) -> dict[str, object]:
        """Sends one message with a fresh request id and returns the data of its reply.

        service is a Block, a unique name, or the scheduler itself for its own endpoints (GraphGRC and the rest). An
        error reply raises Error with its text, and no reply within timeout seconds raises Timeout; a reply that
        arrives later is dropped. The graph must be running: a scheduler answers messages only then. A payload for the
        Settings or StagedSettings endpoint of a Block converts to the block's setting types, as staging converts it;
        sent to a unique name, it takes the types its Python values select.
        """
        name = _service_name(service)
        if not self._in_run():
            raise SchedulerError(f"a request to '{name}' needs a running graph", "Scheduler.request")
        request_id = _fresh_id("request")
        reply = _Reply()
        with self._lock:
            self._pending[request_id] = reply
        try:
            self._send(command, name, endpoint, payload, request_id, _payload_types(service, endpoint, payload))
            deadline = time.monotonic() + timeout
            pause = 0.0005
            while True:
                self._drain()
                if reply.done.is_set():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self._lock:
                        self._expired[request_id] = None
                        if len(self._expired) > EXPIRED_REQUESTS:
                            del self._expired[next(iter(self._expired))]
                    raise Timeout(f"no reply from '{name}' on '{endpoint}' within {timeout} s", "Scheduler.request")
                reply.done.wait(min(pause, remaining))
                pause = min(2 * pause, DISPATCH_INTERVAL)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        if reply.event.is_error:
            raise Error(reply.event.text, reply.event.source, reply.event.time_ns)
        return reply.event.data

    def subscribe(
        self,
        service: "str | Block | Scheduler",
        endpoint: str,
        callback: Callable[[Event], object] | None = None,
    ) -> Subscription:
        """Subscribes to the notifications service makes on endpoint and returns the Subscription.

        A callback runs on the scheduler's dispatch thread, once per notification, in order; an exception it raises is
        reported through sys.excepthook. Without a callback, the notifications go to the queue events() reads.
        """
        subscription = Subscription(self, _service_name(service), endpoint, callback, _fresh_id("subscription"))
        with self._lock:
            self._subscriptions[subscription.request_id] = subscription
        try:
            self._send(Command.SUBSCRIBE, subscription.service, endpoint, None, subscription.request_id)
        except BaseException:
            with self._lock:
                del self._subscriptions[subscription.request_id]
            raise
        if callback is not None:
            self._start_dispatcher()
        return subscription

    def _unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if self._subscriptions.pop(subscription.request_id, None) is None:
                return
            subscription._active = False
        if not self._closed:
            self._send(Command.UNSUBSCRIBE, subscription.service, subscription.endpoint, None, subscription.request_id)
        self._stop_dispatcher(None)

    def _start_dispatcher(self) -> None:
        with self._lock:
            if self._dispatcher is not None:
                return
            self._dispatch_stop = threading.Event()
            self._dispatcher = threading.Thread(
                target=self._dispatch, args=(self._dispatch_stop,), name="gr4-dispatch", daemon=True
            )
            self._dispatcher.start()

    def _stop_dispatcher(self, timeout: float | None) -> None:
        """Stops the dispatch thread when no callback subscription is left, and waits for it.

        The wait lasts at most timeout seconds, without limit when timeout is None. A thread still in a callback then
        is left to finish that callback, and a RuntimeWarning reports it.
        """
        with self._lock:
            thread = self._dispatcher
            if thread is None or any(other.callback is not None for other in self._subscriptions.values()):
                return
            self._dispatcher = None
            self._dispatch_stop.set()
        if thread is threading.current_thread():
            return
        thread.join(timeout)
        if thread.is_alive():
            warnings.warn(
                f"the dispatch thread of {self!r} is still in a callback after {timeout} s; it is left to finish",
                RuntimeWarning,
                stacklevel=3,
            )

    def _dispatch(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self._drain()
            self._deliver(stop)
            stop.wait(DISPATCH_INTERVAL)

    def _deliver(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                subscription, event = self._deliveries.get_nowait()
            except queue.Empty:
                return
            if subscription.active:
                try:
                    subscription.callback(event)
                except Exception:
                    sys.excepthook(*sys.exc_info())

    def close(self) -> None:
        """Stops the graph, ends the subscriptions and the dispatch thread, and releases the scheduler.

        The Graph stays readable and cannot run again: a structural edit raises SchedulerError, and so do run() and a
        new Scheduler of it. A callback still running after CLOSE_TIMEOUT seconds is left to finish, and a
        RuntimeWarning reports it. A second call does nothing.
        """
        if self._closed:
            return
        with self._lock:
            for subscription in self._subscriptions.values():
                subscription._active = False
            self._subscriptions.clear()
        self._stop_dispatcher(CLOSE_TIMEOUT)
        with self._lock:
            self._closed = True
        self._runtime.close()
        _live.discard(self)

    def __repr__(self) -> str:
        """Names the scheduler type and its state."""
        state = "closed" if self._closed else self.state.name
        return f"<Scheduler {self.type} {state}>"


@atexit.register
def _close_live_schedulers() -> None:
    """Closes every scheduler still open when the interpreter exits."""
    for scheduler in list(_live):
        scheduler.close()
