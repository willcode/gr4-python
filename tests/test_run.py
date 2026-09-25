"""Tests of a graph of plugin blocks run to completion by a runtime."""

import os
import sys
import tempfile
import unittest

from gr4 import _gr4

COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"
FILE_SINK = "gr::blocks::fileio::BasicFileSink<float32>"
FILE_SOURCE = "gr::blocks::fileio::BasicFileSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
COPY = "gr::blocks::testing::Copy<float32>"
FLOAT32_BYTES = 4
SAMPLES = 1000
# The longest a run of SAMPLES may take before a case fails.
RUN_TIMEOUT = 30.0


def counted_chain(path, samples):
    """Returns a graph of a counting source, a head block of samples and a file sink writing to path."""
    graph = _gr4.RuntimeGraph()
    source = graph.emplace(COUNTING_SOURCE, "source")
    head = graph.emplace(HEAD, "head", {"n_samples_max": samples}, {"n_samples_max": "uint32"})
    sink = graph.emplace(FILE_SINK, "sink", {"file_name": path})
    graph.connect(source, "out", head, "in")
    graph.connect(head, "out", sink, "in")
    return graph


class RunToCompletion(unittest.TestCase):
    """The blocks come from the plugin libraries, opened by the loader alone, with no preload."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def test_a_head_block_of_n_samples_leaves_a_file_of_n_samples(self):
        for samples in (1000, 65536 + 17):
            with self.subTest(samples=samples):
                path = os.path.join(self.directory, f"counted-{samples}.f32")
                runtime = _gr4.Runtime.create(counted_chain(path, samples))
                runtime.run_and_wait()
                runtime.close()
                self.assertEqual(os.path.getsize(path), samples * FLOAT32_BYTES)

    def test_a_graph_handed_to_create_moves_into_the_scheduler(self):
        graph = counted_chain(os.path.join(self.directory, "moved.f32"), 10)
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)
        with self.assertRaisesRegex(_gr4.Error, "moved into a scheduler"):
            graph.blocks()
        self.assertEqual({block.name for block in runtime.graph().blocks()}, {"source", "head", "sink"})

    def test_an_unknown_scheduler_type_leaves_the_graph_in_place(self):
        graph = counted_chain(os.path.join(self.directory, "kept.f32"), 10)
        with self.assertRaisesRegex(_gr4.Error, "gr::scheduler::NoSuchScheduler"):
            _gr4.Runtime.create(graph, "gr::scheduler::NoSuchScheduler")
        self.assertEqual(len(graph.blocks()), 3)

    def test_a_view_is_refused_and_stays_usable(self):
        graph = _gr4.RuntimeGraph()
        interior = graph.interior(graph.emplace_subgraph("inner"))
        copy = interior.emplace(COPY, "copy")
        with self.assertRaisesRegex(_gr4.Error, "view"):
            _gr4.Runtime.create(interior)
        self.assertEqual(interior.blocks(), [copy])

    def test_a_closed_runtime_refuses_calls_and_closes_again(self):
        runtime = _gr4.Runtime.create(counted_chain(os.path.join(self.directory, "closed.f32"), 10))
        runtime.close()
        with self.assertRaisesRegex(_gr4.Error, "closed"):
            runtime.run_and_wait()
        runtime.close()

    def test_request_state_moves_the_scheduler_and_refuses_an_invalid_transition(self):
        path = os.path.join(self.directory, "requested.f32")
        runtime = _gr4.Runtime.create(counted_chain(path, 10))
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.state, _gr4.State.IDLE)
        runtime.request_state(_gr4.State.INITIALIZED)
        self.assertEqual(runtime.state, _gr4.State.INITIALIZED)
        with self.assertRaisesRegex(_gr4.Error, "PAUSED"):
            runtime.request_state(_gr4.State.PAUSED)
        self.assertEqual(runtime.state, _gr4.State.INITIALIZED)
        runtime.run_and_wait()
        self.assertEqual(os.path.getsize(path), 10 * FLOAT32_BYTES)


class Stops(unittest.TestCase):
    """stop() without a run in progress touches nothing: before the first run and after a run ended by itself."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def test_a_stop_before_any_run_leaves_idle_and_the_next_run_writes_all_n_samples(self):
        for begin in (_gr4.Runtime.run_and_wait, _gr4.Runtime.start):
            with self.subTest(begin=begin.__name__):
                path = os.path.join(self.directory, f"{begin.__name__}.f32")
                runtime = _gr4.Runtime.create(counted_chain(path, SAMPLES))
                self.addCleanup(runtime.close)
                runtime.stop()
                self.assertEqual(runtime.state, _gr4.State.IDLE)
                begin(runtime)
                self.assertTrue(runtime.wait(RUN_TIMEOUT))
                self.assertIsNone(runtime.result())
                self.assertEqual(runtime.state, _gr4.State.STOPPED)
                self.assertEqual(os.path.getsize(path), SAMPLES * FLOAT32_BYTES)

    def test_a_stop_after_a_run_ended_by_itself_leaves_the_state_and_the_result(self):
        path = os.path.join(self.directory, "ended.f32")
        runtime = _gr4.Runtime.create(counted_chain(path, SAMPLES))
        self.addCleanup(runtime.close)
        runtime.run_in_thread()
        self.assertTrue(runtime.wait(RUN_TIMEOUT))
        for stop in range(2):
            with self.subTest(stop=stop):
                runtime.stop()
                self.assertEqual(runtime.state, _gr4.State.STOPPED)
                self.assertIsNone(runtime.result())
        self.assertEqual(os.path.getsize(path), SAMPLES * FLOAT32_BYTES)
        runtime.close()
        self.assertIsNone(runtime.result())


class FailedRun(unittest.TestCase):
    """A run that fails raises its Error from result() and from run_and_wait()."""

    def test_a_file_source_without_its_file_fails_the_run(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        missing = os.path.join(directory.name, "missing.f32")
        graph = _gr4.RuntimeGraph()
        source = graph.emplace(FILE_SOURCE, "source", {"file_name": missing})
        sink = graph.emplace(NULL_SINK, "sink")
        graph.connect(source, "out", sink, "in")
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)

        runtime.run_in_thread()
        self.assertTrue(runtime.wait(60.0))
        with self.assertRaises(_gr4.Error) as caught:
            runtime.result()
        self.assertEqual(runtime.state, _gr4.State.ERROR)
        reported = [event.text for event in runtime.poll_events(1024) if event.is_error]
        self.assertTrue(any(missing in text for text in reported), reported)
        with self.assertRaises(_gr4.Error) as again:
            runtime.run_and_wait()
        self.assertEqual(again.exception.message, caught.exception.message)
        self.assertFalse(os.path.exists(missing))
        runtime.close()
        with self.assertRaises(_gr4.Error) as closed:
            runtime.result()
        self.assertEqual(closed.exception.message, caught.exception.message)


class Defaults(unittest.TestCase):
    """reset_defaults() restores the settings a block was built with, and store_defaults() makes the active ones
    the defaults."""

    def test_a_run_applies_staged_settings_that_the_defaults_restore_and_store(self):
        graph = _gr4.RuntimeGraph()
        source = graph.emplace(COUNTING_SOURCE, "source")
        head = graph.emplace(HEAD, "head", {"n_samples_max": 100}, {"n_samples_max": "uint32"})
        sink = graph.emplace(NULL_SINK, "sink")
        graph.connect(source, "out", head, "in")
        graph.connect(head, "out", sink, "in")
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)
        self.assertEqual(head.default_parameters()["n_samples_max"], 100)

        head.set_staged({"n_samples_max": 7}, {"n_samples_max": "uint32"})
        runtime.run_and_wait()
        self.assertEqual(head.get_value("n_samples_max"), 7)
        self.assertEqual(head.default_parameters()["n_samples_max"], 100)
        head.reset_defaults()
        self.assertEqual(head.get_value("n_samples_max"), 100)

        head.set_staged({"n_samples_max": 11}, {"n_samples_max": "uint32"})
        runtime.run_and_wait()
        self.assertEqual(head.get_value("n_samples_max"), 11)
        head.store_defaults()
        self.assertEqual(head.default_parameters()["n_samples_max"], 11)


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
