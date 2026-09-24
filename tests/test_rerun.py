"""Tests of a stopped runtime running its graph a second time."""

import os
import sys
import tempfile
import time
import unittest

from gr4 import _gr4

COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"
FILE_SINK = "gr::blocks::fileio::BasicFileSink<float32>"
NULL_SOURCE = "gr::blocks::testing::NullSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
FLOAT32_BYTES = 4
SAMPLES = 1000
# The longest a run may take to write its samples before a case fails.
WRITE_TIMEOUT = 30.0


def chain_beside_an_endless_one(path):
    """Returns a graph of a counting source, a head block of SAMPLES and a file sink writing to path, beside a null
    source into a null sink that keeps the run going until it is stopped; and the file sink and the null sink."""
    graph = _gr4.RuntimeGraph()
    source = graph.emplace(COUNTING_SOURCE, "source")
    head = graph.emplace(HEAD, "head", {"n_samples_max": SAMPLES}, {"n_samples_max": "uint32"})
    sink = graph.emplace(FILE_SINK, "sink", {"file_name": path})
    graph.connect(source, "out", head, "in")
    graph.connect(head, "out", sink, "in")
    endless_source = graph.emplace(NULL_SOURCE, "endless_source")
    endless_sink = graph.emplace(NULL_SINK, "endless_sink")
    graph.connect(endless_source, "out", endless_sink, "in")
    return graph, sink, endless_sink


def until_written(path, size):
    """Waits until the file at path holds size bytes, at most WRITE_TIMEOUT seconds, and returns the size it holds."""
    deadline = time.monotonic() + WRITE_TIMEOUT
    while (not os.path.exists(path) or os.path.getsize(path) < size) and time.monotonic() < deadline:
        time.sleep(0.001)
    return os.path.getsize(path) if os.path.exists(path) else 0


class SecondRun(unittest.TestCase):
    """runAndWait() accepts a scheduler in the stopped state, and the blocks run again."""

    def test_a_stopped_runtime_runs_its_graph_again_into_a_new_file(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        first = os.path.join(directory.name, "first.f32")
        second = os.path.join(directory.name, "second.f32")

        graph = _gr4.RuntimeGraph()
        source = graph.emplace(COUNTING_SOURCE, "source")
        head = graph.emplace(HEAD, "head", {"n_samples_max": SAMPLES}, {"n_samples_max": "uint32"})
        sink = graph.emplace(FILE_SINK, "sink", {"file_name": first})
        graph.connect(source, "out", head, "in")
        graph.connect(head, "out", sink, "in")
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)

        runtime.run_and_wait()
        self.assertEqual(os.path.getsize(first), SAMPLES * FLOAT32_BYTES)
        self.assertEqual(runtime.state, _gr4.State.STOPPED)

        self.assertEqual(sink.set_staged({"file_name": second}), {})
        runtime.run_and_wait()
        self.assertTrue(os.path.exists(second))
        self.assertEqual(os.path.getsize(second), SAMPLES * FLOAT32_BYTES)


class StopThenRunAgain(unittest.TestCase):
    """A run stopped while it runs leaves the runtime ready for the next run, and that run processes samples."""

    def stop_and_run_again(self, begin):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        first = os.path.join(directory.name, "first.f32")
        second = os.path.join(directory.name, "second.f32")
        graph, sink, _ = chain_beside_an_endless_one(first)
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)

        begin(runtime)
        self.assertEqual(until_written(first, SAMPLES * FLOAT32_BYTES), SAMPLES * FLOAT32_BYTES)
        self.assertEqual(runtime.state, _gr4.State.RUNNING)
        runtime.stop()
        self.assertTrue(runtime.wait(0.0))
        self.assertEqual(runtime.state, _gr4.State.STOPPED)

        self.assertEqual(sink.set_staged({"file_name": second}), {})
        begin(runtime)
        self.assertEqual(until_written(second, SAMPLES * FLOAT32_BYTES), SAMPLES * FLOAT32_BYTES)
        self.assertEqual(runtime.state, _gr4.State.RUNNING)
        runtime.stop()
        self.assertIsNone(runtime.result())

    def test_start_stop_start_runs_the_graph_again(self):
        self.stop_and_run_again(_gr4.Runtime.start)

    def test_run_in_thread_stop_run_in_thread_runs_the_graph_again(self):
        self.stop_and_run_again(_gr4.Runtime.run_in_thread)


class PortsWhileRunning(unittest.TestCase):
    """A port reports its edge connected while the scheduler runs, and only then."""

    def test_a_port_is_connected_while_the_scheduler_runs(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "written.f32")
        graph, _, endless_sink = chain_beside_an_endless_one(path)
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)
        view = runtime.graph()
        self.assertFalse(view.input_ports(endless_sink)[0].is_connected)
        runtime.run_in_thread()
        # The written samples show the run under way: the scheduler has connected its edges and is not starting.
        self.assertEqual(until_written(path, SAMPLES * FLOAT32_BYTES), SAMPLES * FLOAT32_BYTES)
        self.assertTrue(view.input_ports(endless_sink)[0].is_connected)
        runtime.stop()
        self.assertTrue(runtime.wait(0.0))
        self.assertFalse(view.input_ports(endless_sink)[0].is_connected)


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
