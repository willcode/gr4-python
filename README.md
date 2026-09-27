# gr4

gr4 is a Python package that builds, inspects, runs and stops GNU Radio 4 flow
graphs and their schedulers. Blocks and schedulers are named by their registry
keys and configured through their reflected settings. Every registered block,
built in or loaded from a plugin, is usable from Python without code of its
own.

gr4 is a control interface. Samples move between blocks inside the graph; file,
network and ZeroMQ blocks carry them in and out of it.

The package has two layers. `gr4._gr4` is a pybind11 extension over the
type-erased runtime of `gnuradio4-core` (`gnuradio-4.0/Runtime.hpp`). The
`gr4` package is pure Python on top of it and is the interface described here.

## Requirements

- An installed `gnuradio4-core` that provides `gnuradio-4.0/Runtime.hpp`,
  built with the block registry enabled, and the block plugins to use with it.
- CMake 3.25 or later, Ninja, and a C++23 compiler of the same family as the
  one that built `gnuradio4-core`.
- Python 3.12 or later with its development files, and pybind11 3.

## Building and installing

Point `CMAKE_PREFIX_PATH` at the prefix that holds `gnuradio4-core`:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/gnuradio4 \
      -DCMAKE_INSTALL_PREFIX=/opt/gnuradio4
cmake --build build
cmake --install build
```

The install puts the package in
`<prefix>/<libdir>/python<X.Y>/site-packages/gr4`. Add that `site-packages`
directory to `PYTHONPATH` when it is not on Python's own path:

```sh
export PYTHONPATH=/opt/gnuradio4/lib64/python3.14/site-packages
python3 -c 'import gr4; print(len(gr4.registry.block_types()))'
```

The build tree also holds an importable package: `PYTHONPATH=build/python`
imports `gr4` from it.

The plugin loader reads the directories in `GNURADIO4_PLUGIN_DIRECTORIES`,
separated by colons. When the variable is unset, gr4 sets it to
`<libdir>/gnuradio-4/plugins` of the `gnuradio4-core` prefix it was built
against. Blocks installed elsewhere are named by the variable, or by
`gr4.configure()` before the first use of the registry:

```python
import gr4

gr4.configure(plugin_dirs=["/opt/gnuradio4/lib64/gnuradio-4/plugins"])
```

## Running the tests

The tests use the standard library's `unittest` and open no hardware. Each
file under `tests/` is one `ctest` entry, run with the build tree's
`PYTHONPATH`. The cache variable `GR4_TEST_PLUGIN_DIRECTORIES` names the
plugin directories the tests load blocks from; it defaults to the core
prefix's own.

```sh
cmake -B build \
      -DGR4_TEST_PLUGIN_DIRECTORIES=/opt/blocks/lib64/gnuradio-4/plugins
ctest --test-dir build --output-on-failure
```

## Finding block types

`gr4.registry` lists what this process can build, and `registry[key]`
describes one type: its settings with their types, defaults, units and
documentation, and its ports. `help()` shows the description.

```python
from gr4 import registry

registry.search("HeadBlock")          # every key that contains the text
key = registry.find("HeadBlock", "float32")
head = registry[key]
head.settings["n_samples_max"].type   # 'uint32'
[port.name for port in head.outputs]  # ['out']
help(head)
```

A description is read from one default-constructed instance, once per process.
`registry.describe_all()` reads many in child processes and reports a type
whose construction crashes as a failure; the interpreter carries on.

## Building and running a graph

`Graph.add()` takes a registry key or a `BlockType`, a name, and construction
settings, which are in force when the block starts. `connect()` takes ports,
or blocks with one stream port in the direction needed; more than two
arguments form a chain. `run()` runs the graph to its end.

```python
import gr4

graph = gr4.Graph()
tone = graph.add("gr::blocks::basic::SignalGenerator<float32>", "tone",
                 sample_rate=48_000, tone_frequency=440)
head = graph.add(gr4.registry.find("HeadBlock", "float32"), "head",
                 n_samples_max=1000)
sink = graph.add("gr::blocks::fileio::BasicFileSink<float32>", "sink",
                 file_name="/tmp/tone.f32")
graph.connect(tone, head, sink)
graph.run()
```

Each keyword is checked against the type's description: a key the type does
not declare raises `KeyError` naming the closest keys, and each value is
converted to its setting's type. `sample_rate=16_777_217` therefore reaches a
`float32` setting, rounded to 16777216.0. `strict=False` skips the check and
the conversion, and `meta=` adds meta-information.

A graph also holds subgraphs, whose interior is a `Graph` of its own, and a
nested scheduler, added by its scheduler key:

```python
inner = graph.add_subgraph("inner")
copy = inner.interior.add("gr::blocks::testing::Copy<float32>", "copy")
inner.interior.export(copy.inputs["in"], "in")
inner.interior.export(copy.outputs["out"], "out")
```

## Settings

`block.settings` is a live mapping. `settings[key]` reads the value in force,
and `settings[key] = value` and `update()` stage values, which the block
applies at its next work call. `dict(block.settings)` is a snapshot, and
`defaults`, `staged` and `types` are the other maps.

```python
tone.settings["tone_frequency"] = 880
tone.settings.update(amplitude=0.5, phase=0.0)
tone.settings.staged                  # {'tone_frequency': 880.0, ...}
```

A value a block reads when it starts belongs in `add()`, or in the settings a
graph document is loaded with.

## Running in the background

`gr4.Scheduler(graph, type=None, **settings)` takes a graph, which moves into
the scheduler; the `Graph` object then views the scheduler's graph and stays
usable for inspection and settings. The settings are the scheduler's own,
which it applies when it is created; staging one raises `SchedulerError`.
`start()` returns at once, and the scheduler is a context manager that stops
the graph on exit and raises `RunError` there for a run that failed.

```python
graph = gr4.Graph()
tone = graph.add("gr::blocks::basic::SignalGenerator<float32>", "tone",
                 sample_rate=48_000)
sink = graph.add("gr::blocks::testing::NullSink<float32>", "sink")
graph.connect(tone, sink)
scheduler = gr4.Scheduler(graph, "gr::scheduler::Simple<multiThreaded>",
                          timeout_ms=50)
with scheduler.start(), tone.subscribe("Settings"):
    tone.settings["tone_frequency"] = 880.0
    for event in scheduler.events(timeout=1.0):
        print(event)
```

`stop()`, `pause()`, `resume()` and `wait(timeout=None)` control the run, and
`state` reports it; `stop()` without a run does nothing. A stopped scheduler
runs its graph again. While a scheduler runs, a structural edit such as
`add()`, `connect()`, `remove()` or `export()` raises `GraphBusyError`. After
`close()` the graph stays readable, and a structural edit or a run raises
`SchedulerError`.

Ctrl-C during `run()` stops the graph, waits for it and raises
`KeyboardInterrupt`. An interpreter that exits with a graph running closes
its schedulers first.

## Events, requests and subscriptions

`scheduler.events()` yields what the graph reported since the last read,
oldest first. `scheduler.errors` keeps every error event of the run, and a
run that ends in error raises `RunError`, which carries those events.

`request()` sends one message to a running graph and returns the data of its
reply. It reaches every endpoint of a block and of the scheduler itself:

```python
with scheduler.start():
    tone.request("Settings")["tone_frequency"]
    scheduler.request(scheduler, "GraphGRC")["value"]   # the document
```

`subscribe()` delivers a block's notifications to a callback, which runs on
the scheduler's dispatch thread, or to `events()` without one:

```python
import threading

applied = threading.Event()

def changed(event):
    print("now", event.data["tone_frequency"])
    applied.set()

with scheduler.start(), tone.subscribe("Settings", changed):
    tone.settings["tone_frequency"] = 1320.0
    applied.wait(timeout=1.0)
```

## Graph documents

`Graph.load()`, `Graph.from_yaml()` and `to_yaml()` read and write graph
documents. The `settings` of a load are merged into the named blocks'
parameters before the document loads, and they are in force when the blocks
start. Each key is checked as `add()` checks it. A replaced value keeps the
type the document wrote for it, and a new one takes its setting's type.

```python
chain = gr4.Graph()
chain.connect(chain.add("gr::blocks::testing::NullSource<float32>", "source"),
              chain.add(gr4.registry.find("HeadBlock", "float32"), "head"),
              chain.add("gr::blocks::testing::NullSink<float32>", "sink"))
with open("chain.grc", "w") as file:
    file.write(chain.to_yaml())
graph = gr4.Graph.load("chain.grc",
                       settings={"head": {"n_samples_max": 4096}})
```

`to_yaml()` on a running graph asks the scheduler for the document.

## Errors

Every failure the runtime reports is a `gr4.Error`, with `message`, `where`
and `time_ns`. The checks of keys and values before a call reaches the runtime
raise `KeyError`, `TypeError` and `OverflowError`:

| Error              | Also a         | Raised for                           |
| ------------------ | -------------- | ------------------------------------ |
| `UnknownBlockType` | `LookupError`  | a key no type of the registry has    |
| `PortError`        | `LookupError`  | a port that cannot be connected      |
| `SettingError`     | `ValueError`   | a setting a block refused            |
| `GraphFileError`   | `ValueError`   | a document that does not load        |
| `GraphBusyError`   |                | a structural edit of a running graph |
| `SchedulerError`   |                | a scheduler that cannot act          |
| `RunError`         |                | a run that ended in error            |
| `Timeout`          | `TimeoutError` | a request without a reply            |
| `KeyError`         |                | a setting key the type lacks         |
| `TypeError`        |                | a value of the wrong kind            |
| `OverflowError`    |                | a value out of its setting's range   |

## Examples

Each example runs against the installed blocks:

- `examples/tone_to_file.py` writes a tone to a file and reads it back.
- `examples/list_registry.py` lists the registry by family.
- `examples/live_settings.py` changes a setting of a running graph and prints
  the events it reports.

## License

MIT; see `LICENSE`.
