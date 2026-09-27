# gr4: Python bindings for GNU Radio 4 — design

## Purpose and scope

`gr4` lets a Python program create, inspect, run, monitor and stop GNU Radio 4 graphs and their schedulers.
Blocks and schedulers are named by their registry keys and configured through their reflected settings, so
there is no per-block Python or C++ code: a block that is registered, built in or loaded from a plugin, is
usable from Python without any change here.

It is a control interface. Moving samples between Python and a running graph is outside it; file, network
and ZeroMQ blocks cover that today.

## Foundation

The bindings sit on the type-erased runtime of `gnuradio4-core` (`gnuradio-4.0/Runtime.hpp`): `RuntimeGraph`
builds and inspects a graph by name, `BlockHandle` is the settings surface of one block, and `Runtime` owns a
scheduler instantiated by name and the subscriber on its message output. That header instantiates no framework
template, and every call returns `std::expected`. The extension is one translation unit: it binds each of those
classes as one Python class, converts values, and merges settings into a graph document. It requires the
willcode fork of `gnuradio4-core` built with `GR_ENABLE_BLOCK_REGISTRY` on; upstream main does not carry the
runtime.

The extension must be built with the same compiler family as the installed `gnuradio-core`, a constraint the
runtime header states for every consumer.

## Layers

- `gr4._gr4`, a pybind11 extension: `RuntimeGraph`, `BlockHandle`, `Runtime` and `RuntimeEvent` as they are,
  the conversion between `pmt::Value` and Python objects, the merge of settings into a graph document, and one
  translator from `RuntimeError` to the Python exceptions below. Every call that can block releases the GIL.
- `gr4`, pure Python: the public interface described here, written only against `gr4._gr4`.

## The Python interface

```python
import gr4

g = gr4.Graph()
src  = g.add("gr::blocks::basic::SignalGenerator<float32>", "tone", sample_rate=48_000, tone_frequency=440)
head = g.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=1000)
sink = g.add("gr::blocks::fileio::BasicFileSink<float32>", "sink", file_name="/tmp/tone.f32")
g.connect(src, head, sink)
g.run()

live = gr4.Graph()
tone = live.add("gr::blocks::basic::SignalGenerator<float32>", "tone", sample_rate=48_000, tone_frequency=440)
live.connect(tone, live.add("gr::blocks::testing::NullSink<float32>", "sink"))
sched = gr4.Scheduler(live, "gr::scheduler::Simple<multiThreaded>", timeout_ms=50)
with sched.start():
    tone.settings["tone_frequency"] = 880.0
    for event in sched.events(timeout=2.0):
        print(event)
```

### Registry

`gr4.registry` answers what this process can build: `block_types()`, `scheduler_types()`, `search(text)`, and
`find(family, *type_parameters)`, which returns the one key whose type parameters match or raises naming those
that exist. `registry[key]` is a `BlockType` with `settings` (type, default, unit, description, documentation
and enumeration choices of each), `inputs`, `outputs` (see Ports) and `doc`, which `help()` renders. A type's
description is read once, from a default-constructed instance in a scratch graph, and cached for the process.
`describe_all()` reads every key, each in a child process, so a block whose construction crashes is reported
rather than ending the interpreter.

### Graph

- `add(type, name, *, meta=None, strict=True, **settings)` takes a key string or a `BlockType`; the settings are
  construction parameters, in force when the block starts.
- `add_subgraph(name)`; a nested scheduler is added like any block, by its scheduler key; `block.interior` is
  the `Graph` inside either, and `export(port, name)` / `unexport(port)` on it publish an interior port on the
  boundary.
- `connect(a, b, *more, buffer_size=None, weight=None, name=None)`: each argument is a `Port`, or a `Block`
  with exactly one stream port in the direction needed; more than two arguments form a chain.
- `disconnect`, `remove`, `find(unique_name)`, iteration over blocks (`recursive=` for nested graphs), `edges`.
- `Graph.load(path_or_file, settings=None)`, `Graph.from_yaml(text, settings=None)` and `to_yaml()`.
  `settings` maps a block name to settings merged into that block's parameters in the document before it
  loads, so they are in force when the block starts, the same point a construction parameter reaches. The merge
  runs in C++ on the parsed document: a replaced value takes the type the document wrote for it, and every other
  value keeps its written type. A document is never round-tripped through Python values, which would widen
  every `!!float32` to a float64 and every `!!complex32` to a complex128 the settings code refuses.
- A graph handed to a `Scheduler` moves into it; the `Graph` object then views the scheduler's graph and stays
  usable for inspection and settings. A structural edit while the scheduler runs is refused with
  `GraphBusyError`.

### Block and Port

- `Block`: `name`, `unique_name`, `type`, `valid`, `inputs` and `outputs`, `settings`, `meta`, `interior`.
- `block.settings` is a live mapping: `[key]` reads the active value, `[key] = value` and `update()` stage, and
  `dict(block.settings)` is a snapshot; `.defaults` and `.staged` are the other two maps.
- `Port`: `name`, `type`, `direction`, `kind` (stream or message), `optional`, `synchronous`, `domain`, and
  `collection` and `index` for an element of a port collection (`in#0` is element 0 of `in`).

### Scheduler

`gr4.Scheduler(graph, type=None, **settings)` instantiates the scheduler by key (the runtime's default when
`type` is None) and takes the graph. `run()` runs to the end and returns; `start()` returns at once and is a
context manager that stops on exit; `stop()`, `pause()`, `resume()`, `wait(timeout=None)`, `state`
(`gr4.State`), `settings` (the scheduler's own; they are set when the scheduler is created, and staging one is
refused, because the core applies a scheduler's staged settings only in its constructor), `events()`, `errors`,
`send()` and `request()`. A stopped scheduler can run again: `runAndWait()` accepts one in the stopped state, and a test
pins that the blocks run a second time. `graph.run(scheduler=None, **settings)` creates one and runs it.

### Events, requests and subscriptions

`gr4.Event` is a frozen record: `source`, `endpoint`, `command` (`gr4.Command`), `request_id`, `is_error`,
`text`, `data`, `time_ns`.

- `scheduler.events(timeout=None)` yields what the graph reported since the last read, oldest first.
- `scheduler.errors` keeps every error event of the run; the runtime's own queue holds 1024 events and drops
  the oldest, so error events are copied out on every drain.
- `scheduler.request(service, endpoint, payload=None, command=Command.GET, timeout=1.0)` sends one message with
  a fresh request id and returns the reply's data or raises; `block.request(endpoint, ...)` addresses the
  block by its unique name. This reaches every endpoint the framework and a block define, including the
  scheduler's own.
- `block.subscribe(endpoint, callback)` returns a `Subscription`, which is a context manager and has
  `cancel()`. Callbacks run on one dispatch thread per scheduler, which exists only while a callback
  subscription does.

### Errors

`gr4.Error` carries `where` and `time_ns`. Under it: `UnknownBlockType` and `PortError` (also `LookupError`),
`SettingError` (also `ValueError`), `GraphFileError` (also `ValueError`; its message is the reader's, which names the line and column),
`GraphBusyError`, `SchedulerError`, `RunError` (with the error events of the run) and `Timeout` (also
`TimeoutError`).

## Values and settings

Python to `pmt::Value` when the target type is unknown: `bool` to Bool, `int` to Int64 (UInt64 above the Int64
range), `float` to Float64, `complex` to Complex128, `str` to String, `dict` to Map, `None` to the empty value,
and a list or tuple to a tensor of type-erased elements, which the settings code converts element by element
with its scalar rule. A numpy scalar or one-dimensional array, when numpy is present, keeps its exact dtype.

When the target type is known — from the block's active values, or from the `BlockType` for `add()` — the value
is first coerced to it in Python, and a failure is a `TypeError` or `OverflowError` naming the key and the type
it needs. That also removes two refusals of the settings code. The settings code converts an integer to a float
only when the float holds it exactly, so `sample_rate=16_777_217` (2**24 + 1) fails on a `float32` setting without
coercion and reaches it as 16777216.0 with it. A complex value never changes precision, so a Python `complex` fails
on a `complex<float32>` setting without coercion.

With `strict=True`, a key the block does not declare raises `KeyError` naming the closest declared keys before
the block is constructed; without it, construction itself refuses the key and names no alternative. Meta-information is
passed on purpose through `meta=`, and `strict=False` skips the description and the coercion.

`pmt::Value` to Python: scalars to Python scalars, tensors to lists, maps to dicts, enumerations as their
strings.

A setting changed after the block joined its graph is staged and applied at the block's next work call, which
comes after the block's `start()`; a value a block reads in `start()` belongs in `add()` or in the `settings`
of `Graph.load()`. The settings `set()` call of the framework only stores a value and a running block never
applies it, so the bindings stage.

## Threads, interrupts and lifetime

- `run()` and `start()` share one mechanism: `Runtime::start()` runs `runAndWait()` on a thread the runtime owns
  and keeps its result, which `wait()` returns or raises as `RunError`.
- `run()` waits in slices with the GIL released; on `KeyboardInterrupt` it stops the graph, waits for the run's
  end and raises again. `stop()` waits for the run's end in the same slices through `Runtime::stopFor()`, and a
  `KeyboardInterrupt` there leaves the stop requested and the run ending.
- Events are drained by every call that waits (`run`, `wait`, `events`, `request`) and by the dispatch thread
  while one exists.
- An `atexit` hook stops and joins every live scheduler before the interpreter and the plugin loader are torn
  down, since the scheduler's destructor waits for its workers.
- A `Block` keeps its block alive; one removed from its graph reports `valid` as False.

## Plugins

`GNURADIO4_PLUGIN_DIRECTORIES` is honored; when it is unset the extension sets it to the plugin directory of
the prefix it was built against, since the loader otherwise searches a path relative to the working directory.
`gr4.configure(plugin_dirs=...)` sets it before the first use and raises after. Whether the loader's
`RTLD_LOCAL` opening is enough inside a Python process, or the libraries must be preloaded with `RTLD_GLOBAL`
as `gnuradio4-control-plane` does, is settled by a test that builds and runs a graph of plugin blocks, and the
preload is kept only if that test needs it.

## Additions to the runtime

Each is one commit that touches only `Runtime.hpp`, `Runtime.cpp`, the runtime's tests and the runtime's
knowledge file, so the set can be reviewed and submitted with the runtime itself.

1. `RuntimeEvent` carries the message's `command` and `clientRequestID`, so a reply is matched to its request
   and a notification to its subscription.
2. `RuntimeGraph::fromYaml(text)` and `RuntimeGraph::toYaml()`, compiled into the library and returning
   `std::expected`: the graph-file reader throws and is inline template code.
3. `Runtime::scheduler()`, a `BlockHandle` for the scheduler block, so its settings are read and staged like a
   block's.
4. `RuntimeGraph::edges()`: each edge's two ends, buffer size, weight and name.
5. `RuntimeGraph::inputPorts(block)` and `outputPorts(block)`: each port's name, value type, kind, optional
   and synchronous flags, domain, and its collection and index; the existing name lists flatten a collection
   and carry nothing else.

## Coverage against `gnuradio4-control-plane`

The control plane reaches the core through its typed headers. Everything it does with them has an equivalent
here:

| the control plane                                                 | here                                                                       |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `loadGrc` of a document it rewrote with `pmt::yaml`               | `Graph.from_yaml`, with `settings` for the rewrite that injects parameters |
| scheduler by name, from a preference list                         | `gr4.Scheduler(graph, type)`, `registry.scheduler_types()`                 |
| scheduler start and stop                                          | `start`, `stop`, `run`, plus `pause`, `resume`, `wait`                     |
| block settings read and written by message, staged or immediate   | `block.settings` directly, or `block.request` on either endpoint           |
| block catalog: settings, meta-information, ports with collections | `registry[key]` and `describe_all()`                                       |
| plugin directories chosen by the server                           | `GNURADIO4_PLUGIN_DIRECTORIES`, `gr4.configure`                            |

Two things it does are not core access and are not repeated: registering typed scheduler aliases (a Python
program names the registered keys instead), and its managed streams, which proxy a sink block's HTTP or
WebSocket output. The bindings add what the control plane lacks: building a graph by name, subgraphs and
nested schedulers, removal and disconnection, subscriptions and error events for the whole run, saving a graph,
and listing its edges.

## Not included

Per-block wrappers; samples in and out of Python; editing a running graph through the scheduler's
`EmplaceBlock`, `RemoveBlock`, `EmplaceEdge` and `RemoveEdge` messages, which the request mechanism already
reaches and which a later interface can wrap; settings contexts; version-pinned creation; thread-pool
configuration.

## Layout, build and tests

```
CMakeLists.txt        find_package(gnuradio4 CONFIG), pybind11, install into the prefix's site-packages
src/_gr4.cpp          the extension
python/gr4/           __init__, registry, graph, scheduler, values, errors; _gr4.pyi; py.typed
tests/                the standard library's unittest, no hardware, each file in seconds
examples/             tone to file, list the registry, change a setting on a running graph
```

The tests pin relationships: a head block of 1000 samples writes 1000 samples, a staged change arrives in the
next settings notification, a stop from another thread ends a run cleanly, an error inside a block reaches
`RunError`, a document survives a load and a save, and a misspelled key is refused.

## Repository conventions

- MIT license, copyright Jeff Long.
- C++ follows the core's comment style: a one-line brief and contract comments above a declaration, none inside a
  function body unless the line cannot be understood without it.
- Python carries docstrings (PEP 257) on every public module, class and function, since `help()` shows them to the
  user; comments follow the same contract style. Formatting is black's, and C++ formatting is the core's
  `.clang-format`.
- US English throughout.

## Phases

1. The five runtime additions, each with its test and the knowledge file.
2. The extension, tested on its own.
3. The `gr4` package, its tests and the examples.
4. Later: an interface over live graph edits.
