"""Control GNU Radio 4 flow graphs from Python.

gr4 builds, inspects, runs and stops GNU Radio 4 graphs and their schedulers. Blocks and schedulers are named by their
registry keys and configured through their reflected settings. Every registered block is usable without code of its
own. The extension module ``gr4._gr4`` binds the core library's type-erased runtime.

    import gr4

    graph = gr4.Graph()
    tone = graph.add("gr::blocks::basic::SignalGenerator<float32>", "tone", sample_rate=48_000)
    head = graph.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=1000)
    sink = graph.add("gr::blocks::fileio::BasicFileSink<float32>", "sink", file_name="/tmp/tone.f32")
    graph.connect(tone, head, sink)
    graph.run()
"""

from . import registry
from .errors import (
    Error,
    GraphBusyError,
    GraphFileError,
    PortError,
    RunError,
    SchedulerError,
    SettingError,
    Timeout,
    UnknownBlockType,
)
from .graph import Block, Edge, Graph
from .registry import DEFAULT_SCHEDULER, BlockType, Port, Setting, configure
from .scheduler import Command, Event, Scheduler, State, Subscription
from .values import Settings

__all__ = [
    "DEFAULT_SCHEDULER",
    "Block",
    "BlockType",
    "Command",
    "Edge",
    "Error",
    "Event",
    "Graph",
    "GraphBusyError",
    "GraphFileError",
    "Port",
    "PortError",
    "RunError",
    "Scheduler",
    "SchedulerError",
    "Setting",
    "SettingError",
    "Settings",
    "State",
    "Subscription",
    "Timeout",
    "UnknownBlockType",
    "configure",
    "registry",
]
