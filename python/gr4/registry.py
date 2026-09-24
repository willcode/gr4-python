"""The block and scheduler types this process can build, and a description of each.

The module is the registry itself: registry[key] is the BlockType of a key, ``key in registry`` tests one, and
iteration yields the key of every block and scheduler type.

The plugin loader reads GNURADIO4_PLUGIN_DIRECTORIES once, at its first use; configure() sets the variable before
that use.
"""

import dataclasses
import functools
import json
import os
import pickle
import select
import signal
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterable, Iterator
from typing import BinaryIO, NamedTuple

from . import _gr4
from .errors import Error, UnknownBlockType
from .values import closest

__all__ = [
    "DEFAULT_SCHEDULER",
    "BlockType",
    "Descriptions",
    "Port",
    "Setting",
    "block_types",
    "configure",
    "describe_all",
    "family_of",
    "find",
    "scheduler_types",
    "search",
    "type_parameters_of",
]

PLUGIN_DIRECTORIES_VARIABLE = "GNURADIO4_PLUGIN_DIRECTORIES"

DEFAULT_SCHEDULER: str = _gr4.DEFAULT_SCHEDULER
"""The registry key of the scheduler a Scheduler instantiates when it is given no type."""

IDENTITY_SETTINGS = ("name", "unique_name")
"""The settings every block carries for its names; a description leaves them out."""

_loader_used = False


def _note_loader_use() -> None:
    """Records that a call reaches the plugin loader, after which configure() raises."""
    global _loader_used
    _loader_used = True


def configure(*, plugin_dirs: str | os.PathLike | Iterable[str | os.PathLike]) -> None:
    """Sets the directories the plugin loader reads blocks from, one directory or several.

    The loader reads GNURADIO4_PLUGIN_DIRECTORIES once, at its first use. configure() comes before any other call of
    gr4 that reaches the registry, a graph or a scheduler, and raises Error after one.
    """
    refusal = Error(
        "configure() comes before the first use of the registry: the plugin loader has read its directories",
        "gr4.configure",
    )
    if _loader_used:
        raise refusal
    directories = [plugin_dirs] if isinstance(plugin_dirs, (str, os.PathLike)) else list(plugin_dirs)
    joined = ":".join(os.fspath(directory) for directory in directories)
    previous = os.environ.get(PLUGIN_DIRECTORIES_VARIABLE)
    os.environ[PLUGIN_DIRECTORIES_VARIABLE] = joined
    # The extension reports the directories the loader read, once it has read them.
    if _gr4.plugin_directories() != [directory for directory in joined.split(":") if directory]:
        if previous is None:
            del os.environ[PLUGIN_DIRECTORIES_VARIABLE]
        else:
            os.environ[PLUGIN_DIRECTORIES_VARIABLE] = previous
        raise refusal


@functools.cache
def _block_keys() -> tuple[str, ...]:
    _note_loader_use()
    return tuple(sorted(_gr4.available_block_types()))


@functools.cache
def _scheduler_keys() -> tuple[str, ...]:
    _note_loader_use()
    return tuple(sorted(_gr4.available_scheduler_types()))


@functools.cache
def _every_key() -> tuple[str, ...]:
    return tuple(sorted(set(_block_keys()) | set(_scheduler_keys())))


@functools.cache
def _key_set() -> frozenset[str]:
    return frozenset(_every_key())


def block_types() -> list[str]:
    """Returns the registry key of every block type, sorted."""
    return list(_block_keys())


def scheduler_types() -> list[str]:
    """Returns the registry key of every scheduler type, sorted."""
    return list(_scheduler_keys())


def search(text: str) -> list[str]:
    """Returns the block and scheduler keys that contain text, ignoring case, sorted."""
    needle = text.lower()
    return [key for key in _every_key() if needle in key.lower()]


def family_of(key: str) -> str:
    """Returns the part of a key before its type parameters: 'gr::blocks::testing::HeadBlock'."""
    return key.split("<", 1)[0]


def type_parameters_of(key: str) -> tuple[str, ...]:
    """Returns the type parameters of a key, split at its top-level commas: ('complex<float32>',)."""
    start = key.find("<")
    if start < 0 or not key.endswith(">"):
        return ()
    parameters, current, depth = [], [], 0
    for character in key[start + 1 : -1]:
        if character in "<(":
            depth += 1
        elif character in ">)":
            depth -= 1
        if character == "," and depth == 0:
            parameters.append("".join(current))
            current = []
        else:
            current.append(character)
    parameters.append("".join(current))
    return tuple(" ".join(parameter.split()) for parameter in parameters)


def _in_family(key: str, family: str) -> bool:
    name = family_of(key)
    return name == family or name.endswith("::" + family)


def _spelled(parameters: tuple[str, ...]) -> str:
    return f"<{', '.join(parameters)}>"


def find(family: str, *type_parameters: str) -> str:
    """Returns the one key of family whose type parameters are type_parameters.

    family is a key's name without its type parameters, qualified ('gr::blocks::testing::HeadBlock') or not
    ('HeadBlock'). With no type parameters the family must hold one key. A family that does not exist raises
    UnknownBlockType naming the closest families, and parameters that select no key, or several, raise it naming the
    type parameters that exist.
    """
    wanted = tuple(" ".join(parameter.split()) for parameter in type_parameters)
    members = [key for key in _every_key() if _in_family(key, family)]
    if not members:
        families = {family_of(key) for key in _every_key()}
        names = families | {name.rsplit("::", 1)[-1] for name in families}
        suggestions = closest(family, names)
        hint = f"; did you mean {', '.join(repr(name) for name in suggestions)}?" if suggestions else ""
        raise UnknownBlockType(f"no block or scheduler family '{family}'{hint}", "registry.find")
    matches = [key for key in members if type_parameters_of(key) == wanted] if wanted else members
    if len(matches) == 1:
        return matches[0]
    existing = ", ".join(_spelled(type_parameters_of(key)) for key in members)
    if not matches:
        raise UnknownBlockType(f"'{family}' has no type {_spelled(wanted)}; it has {existing}", "registry.find")
    raise UnknownBlockType(
        f"'{family}' holds {len(matches)} types, {existing}; type parameters select one", "registry.find"
    )


@dataclasses.dataclass(frozen=True)
class Setting:
    """One setting a block type declares, as its default-constructed instance reports it.

    type is the type code a value converts to, default the value the instance holds, and unit, description and
    documentation the texts the block declares; choices lists the values of an enumeration and is None otherwise.
    """

    name: str
    type: str
    default: object
    unit: str = ""
    description: str = ""
    documentation: str = ""
    choices: tuple[str, ...] | None = None


@dataclasses.dataclass(frozen=True)
class Port:
    """One port of a block, or of a block type.

    type is the value type, direction 'input' or 'output', and kind 'stream' or 'message'. An element of a port
    collection has the name connect() takes ('in#0'), the collection's name in collection and its position in index.
    block is the Block the port belongs to, and None on a BlockType.
    """

    name: str
    type: str
    direction: str
    kind: str
    optional: bool
    synchronous: bool
    domain: str
    collection: str = ""
    index: int = 0
    block: object = dataclasses.field(default=None, repr=False)

    @classmethod
    def from_runtime(cls, port: _gr4.RuntimePort, block: object = None) -> "Port":
        """Returns the Port a RuntimePort of the extension describes, on block."""
        return cls(**_port_fields(port), block=block)


def _port_fields(port: _gr4.RuntimePort) -> dict[str, object]:
    return {
        "name": port.name,
        "type": port.type_name,
        "direction": "input" if port.is_input else "output",
        "kind": "message" if port.is_message else "stream",
        "optional": port.is_optional,
        "synchronous": port.is_synchronous,
        "domain": port.domain,
        "collection": port.collection,
        "index": port.index,
    }


def _read_description(key: str) -> dict[str, object]:
    """Reads the description of key from a default-constructed instance in a scratch graph, as plain values."""
    _note_loader_use()
    graph = _gr4.RuntimeGraph()
    handle = graph.emplace(key, "description")
    defaults = handle.default_parameters()
    meta = handle.meta_information()
    settings = {}
    for name, code in handle.setting_types().items():
        if name in IDENTITY_SETTINGS:
            continue
        choices = meta.get(f"{name}::enum_values")
        settings[name] = {
            "type": code,
            "default": defaults[name] if name in defaults else handle.get_value(name),
            "unit": str(meta.get(f"{name}::unit", "")),
            "description": str(meta.get(f"{name}::description", "")),
            "documentation": str(meta.get(f"{name}::documentation", "")),
            "choices": tuple(str(choice) for choice in choices) if isinstance(choices, list) else None,
        }
    return {
        "doc": str(meta.get("description", "")),
        "settings": settings,
        "inputs": [_port_fields(port) for port in graph.input_ports(handle)],
        "outputs": [_port_fields(port) for port in graph.output_ports(handle)],
    }


class _Documentation:
    """The __doc__ of BlockType: the class docstring on the class, and the rendered description on an instance.

    help() shows the __doc__ of an instance, and help(registry[key]) therefore shows the description of that type.
    """

    def __init__(self, class_doc: str | None) -> None:
        self._class_doc = class_doc

    def __get__(self, instance: object, owner: type | None = None) -> str | None:
        return self._class_doc if instance is None else instance.doc


class BlockType:
    """A block or scheduler type of the registry, and its description.

    The description is read from one default-constructed instance in a scratch graph the first time any part of it
    is asked for, and kept for the life of the process. help() on a BlockType shows it.
    """

    def __init__(self, key: str) -> None:
        """Names the type registered as key; registry[key] returns the one BlockType of each key."""
        self.key = key
        self._settings: dict[str, Setting] | None = None
        self._inputs: tuple[Port, ...] = ()
        self._outputs: tuple[Port, ...] = ()
        self._doc = ""

    @property
    def family(self) -> str:
        """The key without its type parameters."""
        return family_of(self.key)

    @property
    def type_parameters(self) -> tuple[str, ...]:
        """The type parameters of the key."""
        return type_parameters_of(self.key)

    @property
    def is_scheduler(self) -> bool:
        """True for a scheduler type."""
        return self.key in _scheduler_keys()

    @property
    def described(self) -> bool:
        """True once the description has been read."""
        return self._settings is not None

    def _described(self) -> None:
        if self._settings is None:
            self._adopt(_read_description(self.key))

    def _adopt(self, description: dict[str, object]) -> None:
        self._doc = description["doc"]
        self._inputs = tuple(Port(**fields) for fields in description["inputs"])
        self._outputs = tuple(Port(**fields) for fields in description["outputs"])
        self._settings = {name: Setting(name=name, **fields) for name, fields in description["settings"].items()}

    @property
    def settings(self) -> dict[str, Setting]:
        """The settings the type declares, by name; the two name settings every block carries are left out."""
        self._described()
        return dict(self._settings)

    @property
    def inputs(self) -> tuple[Port, ...]:
        """The input ports of a default-constructed instance."""
        self._described()
        return self._inputs

    @property
    def outputs(self) -> tuple[Port, ...]:
        """The output ports of a default-constructed instance."""
        self._described()
        return self._outputs

    @property
    def doc(self) -> str:
        """The description as text: the key, the block's documentation, its settings and its ports."""
        self._described()
        lines = [self.key, ""]
        if self._doc.strip():
            lines += [self._doc.strip(), ""]
        if self._settings:
            lines.append("Settings:")
        for setting in self._settings.values():
            details = [setting.type, f"default {setting.default!r}"] + ([setting.unit] if setting.unit else [])
            lines.append(f"    {setting.name} ({', '.join(details)})")
            if setting.description and setting.description != setting.name:
                lines.append(f"        {setting.description}")
            if setting.documentation:
                lines.append(f"        {setting.documentation}")
            if setting.choices:
                lines.append(f"        one of {', '.join(setting.choices)}")
        for title, ports in (("Inputs", self._inputs), ("Outputs", self._outputs)):
            if ports:
                lines += ["", f"{title}:"]
            for port in ports:
                details = [port.type, port.kind] + (["optional"] if port.optional else [])
                lines.append(f"    {port.name} ({', '.join(details)})")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        """True when both name the same key."""
        return isinstance(other, BlockType) and other.key == self.key

    def __hash__(self) -> int:
        """The hash of the key."""
        return hash(self.key)

    def __repr__(self) -> str:
        """Names the key."""
        return f"<BlockType {self.key}>"


BlockType.__doc__ = _Documentation(BlockType.__doc__)

_types: dict[str, BlockType] = {}


def _type_of(key: str) -> BlockType:
    known = _types.get(key)
    if known is not None:
        return known
    if key not in _key_set():
        suggestions = closest(key, _every_key())
        hint = f"; did you mean {', '.join(repr(name) for name in suggestions)}?" if suggestions else ""
        raise UnknownBlockType(f"no block or scheduler type '{key}'{hint}", "registry")
    return _types.setdefault(key, BlockType(key))


class Descriptions(NamedTuple):
    """What describe_all() read: the described types by key, and the reason each other key was not described."""

    types: dict[str, BlockType]
    failures: dict[str, str]


_CHILD_PROGRAM = "import sys\nfrom gr4 import registry\nregistry._serve_descriptions(int(sys.argv[1]))\n"
"""The program a child process of describe_all() runs; the one argument is the descriptor it reports on."""


def describe_all(keys: Iterable[str] | None = None, *, per_process: int = 16, timeout: float = 60.0) -> Descriptions:
    """Reads the description of every block and scheduler type, or of those keys names, in child processes.

    Each child process describes up to per_process keys in turn. A type whose construction crashes ends only the
    child that describes it. A child that dies, or spends more than timeout seconds on one key, loses the key it was
    on. That key is reported under failures with the reason, and the next child resumes after it. A key described once
    is kept for the process, as registry[key] keeps it.
    """
    wanted = list(dict.fromkeys(_every_key() if keys is None else keys))
    result = Descriptions({}, {})
    pending = []
    for key in wanted:
        if key not in _key_set():
            result.failures[key] = f"no block or scheduler type '{key}'"
        elif _type_of(key).described:
            result.types[key] = _type_of(key)
        else:
            pending.append(key)
    while pending:
        handled, lost, reason = _describe_in_child(pending[:per_process], timeout)
        for key, description, failure in handled:
            if failure is None:
                block_type = _type_of(key)
                if not block_type.described:
                    block_type._adopt(description)
                result.types[key] = block_type
            else:
                result.failures[key] = failure
        done = len(handled)
        if lost is not None:
            result.failures[lost] = reason
            done += 1
        if done == 0:
            raise Error(f"a describing process ended before its first key: {reason}", "registry.describe_all")
        pending = pending[done:]
    return result


def _describe_in_child(
    keys: list[str], timeout: float
) -> tuple[list[tuple[str, dict | None, str | None]], str | None, str]:
    """Describes keys in one child process.

    Returns the keys it finished in order, each with its description or the reason it failed; the key it was on when
    it ended, or None; and the reason it ended early. The child reads the plugin directories the loader of this
    process read. An exception while the child describes, KeyboardInterrupt among them, kills and reaps the child.
    """
    read_end, write_end = os.pipe()
    environment = dict(os.environ)
    directories = _gr4.plugin_directories()
    if directories:
        environment[PLUGIN_DIRECTORIES_VARIABLE] = ":".join(directories)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, environment.get("PYTHONPATH")]))
    handled, current, reason = [], None, ""
    with tempfile.TemporaryFile() as child_errors:
        try:
            child = subprocess.Popen(
                [sys.executable, "-c", _CHILD_PROGRAM, str(write_end)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=child_errors,
                pass_fds=(write_end,),
                env=environment,
            )
        finally:
            os.close(write_end)
        with os.fdopen(read_end, "rb", buffering=0) as reader:
            try:
                try:
                    child.stdin.write(json.dumps(keys).encode())
                    child.stdin.close()
                except BrokenPipeError:
                    pass
                received = b""
                while True:
                    ready, _, _ = select.select([reader], [], [], timeout)
                    if not ready:
                        child.kill()
                        reason = f"no report within {timeout} s"
                        break
                    data = reader.read(65536)
                    if not data:
                        break
                    received += data
                    while len(received) >= 4 and len(received) >= 4 + int.from_bytes(received[:4], "big"):
                        size = int.from_bytes(received[:4], "big")
                        message = pickle.loads(received[4 : 4 + size])
                        received = received[4 + size :]
                        if message[0] == "start":
                            current = message[1]
                        elif message[0] == "done":
                            handled.append((message[1], message[2], None))
                            current = None
                        else:
                            handled.append((message[1], None, message[2]))
                            current = None
                status = child.wait()
            except BaseException:
                child.kill()
                child.wait()
                raise
        if not reason:
            reason = _exit_reason(status, child_errors)
    return handled, current, reason


def _exit_reason(status: int, child_errors: BinaryIO) -> str:
    if status < 0:
        text = f"the describing process ended with {signal.Signals(-status).name}"
    else:
        text = f"the describing process exited with status {status}"
    child_errors.seek(0)
    lines = child_errors.read().decode(errors="replace").strip().splitlines()
    return f"{text}: {lines[-1]}" if lines else text


def _serve_descriptions(descriptor: int) -> None:
    """Describes the keys a JSON list on standard input names and reports each on descriptor, then exits.

    Each report is a pickled tuple after its length in four bytes: ('start', key) before a key, then ('done', key,
    description) or ('failed', key, reason). The process ends with os._exit, which skips the static destruction of
    the core and plugin libraries.
    """
    keys = json.load(sys.stdin)
    with os.fdopen(descriptor, "wb") as out:

        def report(message: tuple) -> None:
            data = pickle.dumps(message)
            out.write(len(data).to_bytes(4, "big") + data)
            out.flush()

        for key in keys:
            report(("start", key))
            try:
                description = _read_description(key)
            except Exception as error:
                report(("failed", key, str(error) or type(error).__name__))
            else:
                report(("done", key, description))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


class _RegistryModule(types.ModuleType):
    """The class of this module: a mapping from registry key to BlockType."""

    def __getitem__(self, key: str) -> BlockType:
        """Returns the BlockType of key. UnknownBlockType, naming the closest keys, when no type has that key."""
        return _type_of(key)

    def __contains__(self, key: object) -> bool:
        """True when key names a block or scheduler type."""
        return key in _key_set()

    def __iter__(self) -> Iterator[str]:
        """Iterates over the key of every block and scheduler type, sorted."""
        return iter(_every_key())

    def __len__(self) -> int:
        """Returns the number of block and scheduler types."""
        return len(_every_key())


sys.modules[__name__].__class__ = _RegistryModule
