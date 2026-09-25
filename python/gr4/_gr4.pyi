"""The type-erased runtime of gnuradio4-core: graphs built by registry name, block settings, schedulers and the events
a running graph reports.

Settings travel as dicts from str keys to values. Without a type code the Python type selects the value type: bool
to Bool, int to Int64 (UInt64 above the Int64 range), float to Float64, complex to complex<float64>, str to String,
None to the empty value, a dict with str keys to a map, a list or tuple to a tensor of type-erased elements, and an
object with the buffer protocol to the typed tensor of its format and shape (a scalar for zero dimensions).

A type code names the exact type a value converts to: bool, int8, int16, int32, int64, uint8, uint16, uint32,
uint64, float32, float64, complex<float32>, complex<float64> or string; tensor<...> around one of these or around
value for a tensor; map; or none. A value out of range, or a fraction for an integer type, raises OverflowError, and
one of the wrong kind TypeError, each naming the key. An integer converts to a floating-point type the way the same
number written as a float does. A type-erased tensor whose elements are all strings has the code tensor<string>.

A value converts back to Python scalars, lists nested by a tensor's extents, dicts and None.
"""

import enum
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Final, TypeAlias

_Value: TypeAlias = bool | int | float | complex | str | None | list["_Value"] | dict[str, "_Value"]
_Settings: TypeAlias = dict[str, _Value]

DEFAULT_SCHEDULER: Final[str]
"""The registry key of the scheduler Runtime.create() instantiates when it is given no type."""

class State(enum.IntEnum):
    """The lifecycle state of a scheduler."""

    IDLE = 0
    INITIALIZED = 1
    RUNNING = 2
    REQUESTED_PAUSE = 3
    PAUSED = 4
    REQUESTED_STOP = 5
    STOPPED = 6
    ERROR = 7

class Command(enum.IntEnum):
    """A command of the message plane."""

    INVALID = 0
    GET = 1
    SET = 2
    PARTIAL = 3
    FINAL = 4
    READY = 5
    DISCONNECT = 6
    SUBSCRIBE = 7
    UNSUBSCRIBE = 8
    NOTIFY = 9
    HEARTBEAT = 10

class Error(Exception):
    """A failure the runtime reported.

    Every error a runtime call returns is raised as Error, and so is a C++ exception a call lets out.
    """

    message: str
    """The reason."""
    where: str
    """The framework's source location of the error, or the failing method."""
    time_ns: int
    """The time the framework reported the error, in nanoseconds since the epoch, or 0."""

    def __init__(self, message: str = "", where: str = "", time_ns: int = 0) -> None:
        """Creates an error with the three fields of the runtime's error."""

class BlockHandle:
    """One block of a graph: its names, its settings and its meta-information.

    A handle keeps its block alive. Two handles are equal when they name the same block.
    """

    @property
    def valid(self) -> bool:
        """True when the handle names a block."""

    @property
    def name(self) -> str:
        """The name the block was given."""

    @property
    def unique_name(self) -> str:
        """The name that identifies the block in its process."""

    @property
    def type_name(self) -> str:
        """The registry key of the block's type."""

    def get(self, keys: Sequence[str] | None = None) -> _Settings:
        """Returns the active settings, all of them or those keys names."""

    def get_value(self, key: str) -> _Value:
        """Returns the active value of one setting. KeyError when the block has no setting of that name."""

    def active_parameters(self) -> _Settings:
        """Returns the settings in force."""

    def staged_parameters(self) -> _Settings:
        """Returns the settings staged for the block's next work call."""

    def default_parameters(self) -> _Settings:
        """Returns the settings reset_defaults() restores."""

    def setting_types(self) -> dict[str, str]:
        """Returns the type code of every active setting, by key."""

    def set_staged(self, parameters: dict[str, object], types: dict[str, str] | None = None) -> _Settings:
        """Stages settings for the block's next work call; types maps a key to its type code.

        Returns the settings that could not be staged.
        """

    def set(self, parameters: dict[str, object], types: dict[str, str] | None = None) -> _Settings:
        """Stores settings without staging them; types maps a key to its type code.

        Returns the settings that could not be stored.
        """

    def meta_information(self) -> _Settings:
        """Returns the block's meta-information."""

    def set_meta_information(self, information: dict[str, object]) -> None:
        """Replaces the block's meta-information."""

    def store_defaults(self) -> None:
        """Makes the active settings the defaults."""

    def reset_defaults(self) -> None:
        """Restores the default settings."""

    def __eq__(self, other: object) -> bool:
        """True when both handles name the same block."""

    def __hash__(self) -> int:
        """The hash of the unique name."""

class RuntimePort:
    """One port of a block, as RuntimeGraph.input_ports() and output_ports() describe it."""

    @property
    def name(self) -> str:
        """The name connect() accepts: 'out', or 'in#1' for element 1 of a collection."""

    @property
    def type_name(self) -> str:
        """The value type, as port_type_name() spells it."""

    @property
    def is_input(self) -> bool:
        """True for an input port."""

    @property
    def is_message(self) -> bool:
        """True for a message port, False for a stream port."""

    @property
    def is_optional(self) -> bool:
        """True when the port may stay unconnected."""

    @property
    def is_synchronous(self) -> bool:
        """True when the port moves samples in step with the block's other synchronous ports."""

    @property
    def is_connected(self) -> bool:
        """True while a running scheduler holds the port's edge connected."""

    @property
    def domain(self) -> str:
        """The compute domain of the port."""

    @property
    def collection(self) -> str:
        """The name of the collection the port belongs to, or ''."""

    @property
    def index(self) -> int:
        """The port's position in its collection, or 0."""

    @property
    def min_samples(self) -> int:
        """The fewest samples the port takes per work call."""

    @property
    def max_samples(self) -> int:
        """The most samples the port takes per work call."""

class RuntimeEdge:
    """One edge of a graph, as RuntimeGraph.edges() lists it."""

    @property
    def source_block(self) -> BlockHandle:
        """The block the edge leaves."""

    @property
    def source_port(self) -> str:
        """The output port the edge leaves."""

    @property
    def destination_block(self) -> BlockHandle:
        """The block the edge enters."""

    @property
    def destination_port(self) -> str:
        """The input port the edge enters."""

    @property
    def min_buffer_size(self) -> int:
        """The smallest buffer the edge asks for, the framework's default when none was given."""

    @property
    def weight(self) -> int:
        """The edge's weight."""

    @property
    def name(self) -> str:
        """The edge's name."""

class RuntimeGraph:
    """A flow graph built by registry name.

    A RuntimeGraph owns its graph, or views one a subgraph block or a scheduler owns. A graph handed to
    Runtime.create moves into the scheduler, and every later call on this object raises Error.

    The constructor, emplace(), emplace_subgraph(), remove(), clear() and from_yaml() release the GIL, and so does the
    deallocation of a graph, a block handle or an edge.
    """

    def __init__(self, settings: dict[str, object] | None = None, types: dict[str, str] | None = None) -> None:
        """Creates an empty graph with the given graph settings."""

    def emplace(
        self, type: str, name: str, parameters: dict[str, object] | None = None, types: dict[str, str] | None = None
    ) -> BlockHandle:
        """Instantiates the block type registered as type, names it and adds it.

        The parameters are in force when the block starts; types maps a key to its type code. A parameter the block does
        not declare or refuses raises Error with the block's message.
        """

    def emplace_subgraph(self, name: str, parameters: dict[str, object] | None = None) -> BlockHandle:
        """Adds a nested graph; interior() opens it for wiring."""

    def interior(self, block: BlockHandle) -> RuntimeGraph:
        """Returns a view of a subgraph's interior, or of a nested scheduler's graph."""

    def blocks(self, recursive: bool = False) -> list[BlockHandle]:
        """Returns the blocks of this level of the graph, or of every level."""

    def find(self, unique_name: str, recursive: bool = True) -> BlockHandle:
        """Returns the block with that unique name."""

    def remove(self, block: BlockHandle) -> None:
        """Removes a block and its edges."""

    def clear(self) -> None:
        """Removes every block and edge."""

    def connect(
        self,
        source: BlockHandle,
        source_port: str,
        destination: BlockHandle,
        destination_port: str,
        min_buffer_size: int = 0,
        weight: int = 0,
        name: str = "",
    ) -> None:
        """Connects an output port to an input port. A min_buffer_size of 0 selects the framework's default."""

    def disconnect(
        self, source: BlockHandle, source_port: str, destination: BlockHandle, destination_port: str
    ) -> None:
        """Removes the edge between two ports."""

    def input_port_names(self, block: BlockHandle) -> list[str]:
        """Returns the names of a block's input ports."""

    def output_port_names(self, block: BlockHandle) -> list[str]:
        """Returns the names of a block's output ports."""

    def port_type_name(self, block: BlockHandle, is_input: bool, port: str) -> str:
        """Returns the value type of one port, or '' when the block has no such port."""

    def input_ports(self, block: BlockHandle) -> list[RuntimePort]:
        """Describes the ports input_port_names() lists, in the same order."""

    def output_ports(self, block: BlockHandle) -> list[RuntimePort]:
        """Describes the ports output_port_names() lists, in the same order."""

    def edges(self) -> list[RuntimeEdge]:
        """Returns the edges of this level of the graph."""

    def export_port(self, inner_block: BlockHandle, is_input: bool, port: str, exported_name: str) -> None:
        """Publishes a port of an interior block on the subgraph, under exported_name.

        Call on the view interior() returns.
        """

    def unexport_port(self, inner_block: BlockHandle, is_input: bool, port: str) -> None:
        """Withdraws a port export_port() published."""

    def exported_input_ports(self) -> dict[str, dict[str, dict[str, str]]]:
        """Returns the exported input ports: block unique name to port name to {'exportedName': name}."""

    def exported_output_ports(self) -> dict[str, dict[str, dict[str, str]]]:
        """Returns the exported output ports: block unique name to port name to {'exportedName': name}."""

    @staticmethod
    def from_yaml(document: str) -> RuntimeGraph:
        """Loads a graph document into a new graph.

        A document that does not parse raises Error naming the line and column.
        """

    def to_yaml(self) -> str:
        """Writes the graph as a graph document."""

class RuntimeEvent:
    """One report the running graph made on the message plane."""

    @property
    def source(self) -> str:
        """The unique name of the reporting block."""

    @property
    def endpoint(self) -> str:
        """The property endpoint it reported on."""

    @property
    def command(self) -> Command:
        """FINAL for the reply to a request, NOTIFY for a notification."""

    @property
    def client_request_id(self) -> str:
        """The id of the request or subscription the event answers, or ''."""

    @property
    def is_error(self) -> bool:
        """True when the event reports an error."""

    @property
    def text(self) -> str:
        """The rendered error text of an error event."""

    @property
    def data(self) -> _Settings:
        """The message payload of an event that is not an error."""

    @property
    def time_ns(self) -> int:
        """The time the error was raised, in nanoseconds since the epoch, or 0."""

class Quiescence:
    """The context manager Runtime.quiesce() returns. The work loop stays quiesced from entry to exit."""

    def __enter__(self) -> Quiescence:
        """Quiesces the work loop."""

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        """Releases the work loop. Returns False. An exception raised in the with block propagates."""

class Runtime:
    """A scheduler holding a graph, and the subscriber on its message plane.

    The runtime runs the graph on a thread it owns, one run at a time, and keeps the last run's result:
    run_and_wait(), start() and run_in_thread() start a run, and each raises Error while a run is in progress. close()
    stops the run, waits for its end and releases the runtime; the destructor does the same.

    create(), run_and_wait(), start(), run_in_thread(), stop(), wait(), request_state(), close() and entering and
    leaving a quiesce() block release the GIL, and so does the deallocation of a runtime. run_and_wait(), wait(), stop()
    and close() run the signal handlers every 100 ms while they wait in the main thread, and Ctrl-C raises
    KeyboardInterrupt from them. A KeyboardInterrupt from stop() or close() leaves the stop requested and the run
    ending, and close() after it waits for the end.
    """

    @staticmethod
    def create(
        graph: RuntimeGraph,
        type: str | None = None,
        settings: dict[str, object] | None = None,
        types: dict[str, str] | None = None,
    ) -> Runtime:
        """Instantiates the scheduler registered as type, the runtime's default when type is None, with the given
        settings, and hands it the graph, which moves into the scheduler.
        """

    def run_and_wait(self) -> None:
        """Runs the graph to its end on the runtime's thread and waits for it.

        Ctrl-C stops the run, waits for its end and raises KeyboardInterrupt, and a second Ctrl-C ends that wait and
        leaves the run ending; a run that fails raises its Error.
        """

    def start(self) -> None:
        """Runs the graph on the runtime's thread, as run_in_thread() does, and returns.

        stop() ends the run; wait() and result() report it.
        """

    def stop(self) -> None:
        """Stops the run in progress and waits for it to end.

        Without a run in progress it does nothing. Ctrl-C raises KeyboardInterrupt and leaves the stop requested and the
        run ending.
        """

    def run_in_thread(self) -> None:
        """Runs the graph on the runtime's thread and returns; wait() and result() report the run."""

    def wait(self, timeout: float | None = None) -> bool:
        """Waits for the last run to end.

        Returns True when it has ended, False when timeout seconds passed first. A NaN timeout raises ValueError;
        Ctrl-C raises KeyboardInterrupt and leaves the run going.
        """

    def result(self) -> None:
        """Returns None when the last run ended normally, and raises its Error otherwise.

        While a run is in progress it raises Error, and after a start whose thread did not start it raises that start's
        Error. After close() it reports the last run before the close.
        """

    @property
    def state(self) -> State:
        """The scheduler's lifecycle state."""

    def request_state(self, state: State) -> None:
        """Requests a lifecycle transition of the scheduler."""

    def poll_events(self, max_events: int = 64) -> list[RuntimeEvent]:
        """Returns what the graph reported since the last call, oldest first, at most max_events of it."""

    def send(
        self,
        command: Command,
        service: str,
        endpoint: str,
        payload: dict[str, object] | None = None,
        client_request_id: str = "",
        types: dict[str, str] | None = None,
    ) -> None:
        """Sends one message to the scheduler's message input, addressed to service (a block's unique name) and
        endpoint; types maps a payload key to its type code.

        A message sent while the input is full is dropped and send() returns normally; a request dropped that way gets
        no reply, and a caller waiting for one times out.
        """

    def quiesce(self) -> Quiescence:
        """Returns a context manager that quiesces the work loop for a live graph edit."""

    def graph(self) -> RuntimeGraph:
        """Returns a view of the graph the scheduler holds."""

    def scheduler(self) -> BlockHandle:
        """Returns the scheduler as a block, for its own settings."""

    def close(self) -> None:
        """Stops the run in progress, waits for its end and releases the runtime.

        A call on a closed runtime does nothing. Ctrl-C raises KeyboardInterrupt, leaves the stop requested and the run
        ending, and keeps the runtime, which a later close() releases.
        """

def available_block_types() -> list[str]:
    """Returns the registry key of every block type this process can build."""

def available_scheduler_types() -> list[str]:
    """Returns the registry key of every scheduler type this process can build."""

def plugin_directories() -> list[str]:
    """Returns the directories the plugin loader reads, from GNURADIO4_PLUGIN_DIRECTORIES as it was at the loader's
    first use.
    """

def merge_parameters(
    document: str, settings: dict[str, dict[str, object]], types: dict[str, dict[str, str]] | None = None
) -> str:
    """Sets settings (block name to key to value) in the block entries of a graph document and returns the document
    written again with its type tags.

    A block entry at any depth of subgraphs is named by its parameters.name or its unique_name, and a subgraph without
    parameters.name by its top-level name. The settings of a subgraph with a scheduler go to its scheduler.parameters;
    a subgraph without a scheduler takes no settings and raises Error. A parameter held there takes the type of the
    value it holds; one not held takes the type code types gives it (block name to key to code), or the type its
    Python value selects. A name no entry carries, or one two entries carry, raises Error; so does a document that
    does not parse, naming the line and column.
    """

def document_block_types(document: str, names: Iterable[str]) -> dict[str, str]:
    """Maps each of names to the type of the block entry of a graph document that carries the name.

    Entries are named as merge_parameters() names them. The type of a block is the id of its entry, and that of a
    subgraph with a scheduler the id of the scheduler; an entry without an id gives ''. A name no entry carries, one
    two entries carry, or one of a subgraph without a scheduler raises Error; so does a document that does not parse,
    naming the line and column.
    """

def convert(obj: object, type_code: str | None = None) -> _Value:
    """Converts obj to a pmt value and back to Python.

    Without type_code the Python type selects the value type. With type_code the value has exactly that type: a value
    out of range, or a fraction for an integer type, raises OverflowError, and one of the wrong kind TypeError.
    """

def type_code(obj: object, type_code: str | None = None) -> str:
    """Returns the type code of the pmt value obj converts to, by the rule convert() applies."""
