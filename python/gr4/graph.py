"""Graphs of blocks built by registry key, their blocks, ports and edges, and graph documents.

A Graph owns its blocks until a Scheduler takes it; from then on the Graph views the scheduler's graph and stays
usable for inspection and settings. A structural edit while the scheduler runs raises GraphBusyError, and one after the
scheduler is closed raises SchedulerError.
"""

import dataclasses
import os
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, TextIO

from . import _gr4, registry
from .errors import (
    Error,
    GraphBusyError,
    GraphFileError,
    PortError,
    SchedulerError,
    SettingError,
    UnknownBlockType,
    translated,
)
from .registry import BlockType, Port
from .values import Settings, coercion_types, unknown_setting

if TYPE_CHECKING:
    from .scheduler import Scheduler, Subscription

__all__ = ["Block", "Edge", "Graph"]


class Block:
    """One block of a graph, or a subgraph or nested scheduler added like a block.

    A Block keeps its block alive; one removed from its graph reports valid as False. Two Blocks are equal when they
    name the same block.
    """

    def __init__(self, graph: "Graph", handle: _gr4.BlockHandle) -> None:
        """Wraps handle, a block of the level of a graph that graph views."""
        self._graph = graph
        self._handle = handle

    @property
    def name(self) -> str:
        """The name the block was given."""
        return self._handle.name

    @property
    def unique_name(self) -> str:
        """The name that identifies the block in its process; requests address a block by it."""
        return self._handle.unique_name

    @property
    def type(self) -> str:
        """The registry key of the block's type."""
        return self._handle.type_name

    @property
    def valid(self) -> bool:
        """True while the block belongs to its graph."""
        return self._handle.valid and self._handle in self._graph._graph.blocks()

    @property
    def inputs(self) -> dict[str, Port]:
        """The input ports, by the name connect() takes.

        While a scheduler runs the graph, read the ports only when it is neither starting nor stopping.
        """
        return {port.name: Port.from_runtime(port, self) for port in self._graph._graph.input_ports(self._handle)}

    @property
    def outputs(self) -> dict[str, Port]:
        """The output ports, by the name connect() takes, with the reading rule of inputs."""
        return {port.name: Port.from_runtime(port, self) for port in self._graph._graph.output_ports(self._handle)}

    @property
    def settings(self) -> Settings:
        """The block's settings: [key] reads the value in force, and [key] = value and update() stage."""
        return Settings(self._handle, f"block '{self.name}'")

    @property
    def meta(self) -> dict[str, object]:
        """The block's meta-information; assigning a dict replaces it."""
        return self._handle.meta_information()

    @meta.setter
    def meta(self, information: Mapping[str, object]) -> None:
        self._handle.set_meta_information(dict(information))

    @property
    def interior(self) -> "Graph":
        """The Graph inside a subgraph or a nested scheduler. Error for any other block."""
        return Graph._view(self._graph._graph.interior(self._handle), self._graph._root)

    def request(
        self,
        endpoint: str,
        payload: Mapping[str, object] | None = None,
        command: _gr4.Command = _gr4.Command.GET,
        timeout: float = 1.0,
    ) -> dict[str, object]:
        """Sends one request to the block and returns the data of its reply; see Scheduler.request()."""
        scheduler = self._graph._scheduler_or_raise("a request")
        return scheduler.request(self, endpoint, payload, command, timeout)

    def subscribe(self, endpoint: str, callback=None) -> "Subscription":
        """Subscribes to the block's notifications on endpoint; see Scheduler.subscribe()."""
        return self._graph._scheduler_or_raise("a subscription").subscribe(self, endpoint, callback)

    def __eq__(self, other: object) -> bool:
        """True when both name the same block."""
        return isinstance(other, Block) and other._handle == self._handle

    def __hash__(self) -> int:
        """The hash of the unique name."""
        return hash(self._handle)

    def __repr__(self) -> str:
        """Names the block and its type."""
        return f"<Block '{self.name}' {self.type}>"


@dataclasses.dataclass(frozen=True)
class Edge:
    """One edge of a graph: its two ends, the minimum buffer size it holds, its weight and its name."""

    source: Block
    source_port: str
    destination: Block
    destination_port: str
    buffer_size: int
    weight: int
    name: str


def _read_document(path_or_file: str | os.PathLike | TextIO) -> str:
    if hasattr(path_or_file, "read"):
        return path_or_file.read()
    with open(path_or_file, encoding="utf-8") as document:
        return document.read()


class Graph:
    """A flow graph built by registry key.

    add() instantiates blocks, connect() joins their ports, and run() or a Scheduler runs the graph. A graph handed to a
    Scheduler moves into it: the Graph then views the scheduler's graph. The structural edits, add(), add_subgraph(),
    connect(), disconnect(), remove(), export() and unexport(), raise GraphBusyError while the scheduler runs and
    SchedulerError once it is closed.
    """

    def __init__(self) -> None:
        """Creates an empty graph."""
        registry._note_loader_use()
        self._graph = _gr4.RuntimeGraph()
        self._root = self
        self._scheduler: "Scheduler | None" = None

    @classmethod
    def _view(cls, runtime_graph: _gr4.RuntimeGraph, root: "Graph") -> "Graph":
        view = cls.__new__(cls)
        view._graph = runtime_graph
        view._root = root
        view._scheduler = None
        return view

    @classmethod
    def from_yaml(cls, text: str, settings: Mapping[str, Mapping[str, object]] | None = None) -> "Graph":
        """Loads a graph document.

        settings maps a block name to settings merged into that block's parameters in the document before it loads;
        they are in force when the block starts. Each key is checked against the description of the type the document
        gives the block, as add() checks it: a key the type does not declare raises KeyError naming the closest
        declared keys. A replaced value takes the type the document wrote for it, and a new one the type its setting
        declares. A document that does not load raises GraphFileError; a parse failure names the line and column.
        """
        if settings:
            given = {name: dict(values) for name, values in settings.items()}
            with translated(GraphFileError):
                type_keys = _gr4.document_block_types(text, list(given))
            types = {}
            for name, values in given.items():
                key = type_keys[name]
                if key not in registry:
                    continue
                declared = {setting.name: setting.type for setting in registry[key].settings.values()}
                for setting in values:
                    if setting not in declared:
                        raise unknown_setting(setting, declared, f"block '{name}' ({key})")
                types[name] = coercion_types(declared, values)
            with translated(GraphFileError):
                text = _gr4.merge_parameters(text, given, types)
        registry._note_loader_use()
        graph = cls.__new__(cls)
        with translated(GraphFileError):
            graph._graph = _gr4.RuntimeGraph.from_yaml(text)
        graph._root = graph
        graph._scheduler = None
        return graph

    @classmethod
    def load(
        cls, path_or_file: str | os.PathLike | TextIO, settings: Mapping[str, Mapping[str, object]] | None = None
    ) -> "Graph":
        """Loads the graph document a path names or an open file holds; settings as for from_yaml()."""
        return cls.from_yaml(_read_document(path_or_file), settings)

    def to_yaml(self) -> str:
        """Writes the graph as a graph document.

        On a graph a scheduler is running, the document is the reply of the scheduler's GraphGRC endpoint, which
        describes the whole graph; the interior of a block of a running graph raises GraphBusyError.
        """
        scheduler = self._root._scheduler
        if scheduler is not None and scheduler._in_run():
            if self._root is not self:
                raise GraphBusyError("the document of a running graph is the whole graph's", "Graph.to_yaml")
            return scheduler.request(scheduler, "GraphGRC")["value"]
        return self._graph.to_yaml()

    def _scheduler_or_raise(self, what: str) -> "Scheduler":
        scheduler = self._root._scheduler
        if scheduler is None:
            raise SchedulerError(f"{what} needs the graph to belong to a Scheduler", "Graph")
        return scheduler

    def _check_idle(self, operation: str) -> None:
        scheduler = self._root._scheduler
        if scheduler is None:
            return
        if scheduler._closed:
            raise SchedulerError(f"{operation}() is a structural edit, and the graph's scheduler is closed", "Graph")
        if scheduler._in_run():
            raise GraphBusyError(f"{operation}() is a structural edit, and the graph's scheduler is running", "Graph")

    def _block(self, handle: _gr4.BlockHandle) -> Block:
        return Block(self, handle)

    def add(
        self,
        type: str | BlockType,
        name: str,
        *,
        meta: Mapping[str, object] | None = None,
        strict: bool = True,
        **settings: object,
    ) -> Block:
        """Adds a block of the type a registry key or a BlockType names, and returns it.

        The settings are construction parameters, in force when the block starts. A scheduler key adds a nested
        scheduler. With strict, each setting is checked against the type's description: a key the type does not
        declare raises KeyError naming the closest declared keys, and each value is converted to its setting's type,
        raising TypeError or OverflowError when it does not fit. strict=False passes the settings as they are. A
        setting the block refuses raises SettingError with the block's message. meta is merged into the block's
        meta-information.
        """
        self._check_idle("add")
        key = type.key if isinstance(type, BlockType) else type
        types = None
        if strict:
            declared = {setting.name: setting.type for setting in registry[key].settings.values()}
            for setting in settings:
                if setting not in declared:
                    raise unknown_setting(setting, declared, key)
            types = coercion_types(declared, settings)
        registry._note_loader_use()
        try:
            handle = self._graph.emplace(key, name, settings, types)
        except Error as error:
            if error.__class__ is not Error:
                raise
            kind = SettingError if key in registry else UnknownBlockType
            raise kind(error.message, error.where, error.time_ns).with_traceback(error.__traceback__) from None
        if meta:
            handle.set_meta_information({**handle.meta_information(), **meta})
        return self._block(handle)

    def add_subgraph(self, name: str) -> Block:
        """Adds an empty nested graph and returns it; its interior is the Graph to add blocks to."""
        self._check_idle("add_subgraph")
        return self._block(self._graph.emplace_subgraph(name))

    def _end(self, end: Port | Block, direction: str) -> Port:
        if isinstance(end, Block):
            ports = end.outputs if direction == "output" else end.inputs
            streams = [port for port in ports.values() if port.kind == "stream"]
            if len(streams) != 1:
                names = ", ".join(port.name for port in streams) or "none"
                raise PortError(
                    f"block '{end.name}' has {len(streams)} {direction} stream ports ({names}); pass the Port",
                    "Graph.connect",
                )
            return streams[0]
        if not isinstance(end, Port) or not isinstance(end.block, Block):
            raise TypeError(f"a connection end is a Port of a Block or a Block, got {end!r}")
        if end.direction != direction:
            raise PortError(f"port '{end.name}' of '{end.block.name}' is an {end.direction}", "Graph.connect")
        return end

    def _check_member(self, port: Port) -> None:
        if port.block._handle not in self._graph.blocks():
            raise PortError(f"block '{port.block.name}' is not a block of this level of the graph", "Graph.connect")

    def connect(
        self,
        source: Port | Block,
        destination: Port | Block,
        *more: Port | Block,
        buffer_size: int | None = None,
        weight: int | None = None,
        name: str | None = None,
    ) -> None:
        """Connects an output port to an input port, or a chain of more than two ends in turn.

        Each end is a Port, or a Block with exactly one stream port in the direction needed; a Block inside a chain
        is the destination of one edge and the source of the next. The two ports of an edge carry the same value type,
        and a pair of different types raises PortError naming both. Every end and every pair is checked before any
        edge is made. buffer_size is the minimum buffer the edge holds, the framework's default when None.
        """
        self._check_idle("connect")
        chain = [source, destination, *more]
        pairs = [(self._end(chain[i], "output"), self._end(chain[i + 1], "input")) for i in range(len(chain) - 1)]
        for pair in pairs:
            for port in pair:
                self._check_member(port)
        for out, into in pairs:
            if out.type != into.type:
                raise PortError(
                    f"output '{out.name}' of '{out.block.name}' carries {out.type}, and input '{into.name}' of "
                    f"'{into.block.name}' takes {into.type}",
                    "Graph.connect",
                )
        for out, into in pairs:
            with translated(PortError):
                self._graph.connect(
                    out.block._handle,
                    out.name,
                    into.block._handle,
                    into.name,
                    buffer_size or 0,
                    weight or 0,
                    name or "",
                )

    def disconnect(self, source: Port | Block, destination: Port | Block) -> None:
        """Removes the edge between two ends, each given as connect() takes it."""
        self._check_idle("disconnect")
        out, into = self._end(source, "output"), self._end(destination, "input")
        with translated(PortError):
            self._graph.disconnect(out.block._handle, out.name, into.block._handle, into.name)

    def remove(self, block: Block) -> None:
        """Removes a block and its edges; the Block then reports valid as False."""
        self._check_idle("remove")
        self._graph.remove(block._handle)

    def export(self, port: Port, name: str) -> None:
        """Publishes a port of a block of this interior on the subgraph that holds it, under name."""
        self._check_idle("export")
        with translated(PortError):
            self._graph.export_port(port.block._handle, port.direction == "input", port.name, name)

    def unexport(self, port: Port) -> None:
        """Withdraws a port export() published."""
        self._check_idle("unexport")
        with translated(PortError):
            self._graph.unexport_port(port.block._handle, port.direction == "input", port.name)

    def blocks(self, recursive: bool = False) -> list[Block]:
        """Returns the blocks of this level of the graph, or with recursive those of every nested level too."""
        found = [self._block(handle) for handle in self._graph.blocks()]
        if recursive:
            for block in list(found):
                interior = self._interior_of(block)
                if interior is not None:
                    found += interior.blocks(recursive=True)
        return found

    def _interior_of(self, block: Block) -> "Graph | None":
        try:
            return block.interior
        except Error:
            return None

    def find(self, unique_name: str, recursive: bool = True) -> Block:
        """Returns the block with that unique name, looked for at every level with recursive. KeyError if none."""
        for block in self.blocks(recursive):
            if block.unique_name == unique_name:
                return block
        raise KeyError(f"no block with the unique name '{unique_name}'")

    @property
    def edges(self) -> list[Edge]:
        """The edges of this level of the graph."""
        return [
            Edge(
                self._block(edge.source_block),
                edge.source_port,
                self._block(edge.destination_block),
                edge.destination_port,
                edge.min_buffer_size,
                edge.weight,
                edge.name,
            )
            for edge in self._graph.edges()
        ]

    @property
    def scheduler(self) -> "Scheduler | None":
        """The Scheduler the graph moved into, or None."""
        return self._root._scheduler

    def run(self, scheduler: str | None = None, **settings: object) -> None:
        """Runs the graph to the end.

        The first run creates a Scheduler of the type scheduler names, the default when None, with settings; a later
        run runs that scheduler again and takes neither argument. See Scheduler.run().
        """
        from .scheduler import Scheduler

        if self._root is not self:
            raise SchedulerError("run() runs a whole graph, not the interior of a block", "Graph.run")
        if self._scheduler is None:
            Scheduler(self, scheduler, **settings)
        elif scheduler is not None or settings:
            raise SchedulerError("the graph belongs to a scheduler already; run it without arguments", "Graph.run")
        self._scheduler.run()

    def __iter__(self) -> Iterator[Block]:
        """Iterates over the blocks of this level of the graph."""
        return iter(self.blocks())

    def __len__(self) -> int:
        """Returns the number of blocks at this level of the graph."""
        return len(self._graph.blocks())

    def __contains__(self, block: object) -> bool:
        """True when block is a Block of this level of the graph."""
        return isinstance(block, Block) and block._handle in self._graph.blocks()

    def __repr__(self) -> str:
        """Names the number of blocks, and the scheduler when the graph belongs to one."""
        where = f" in {self._root._scheduler!r}" if self._root._scheduler is not None else ""
        return f"<Graph of {len(self)} blocks{where}>"
