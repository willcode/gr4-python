"""Tests of the errors the runtime reports and of the exception that carries them."""

import os
import sys
import unittest

from gr4 import _gr4

COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"


class ErrorClass(unittest.TestCase):
    """Error is an Exception with the three fields of the runtime's error."""

    def test_error_carries_message_where_and_time(self):
        error = _gr4.Error("no such port", "RuntimeGraph::connect", 5)
        self.assertIsInstance(error, Exception)
        self.assertEqual((error.message, error.where, error.time_ns), ("no such port", "RuntimeGraph::connect", 5))
        self.assertEqual(str(error), "no such port")
        bare = _gr4.Error("reason")
        self.assertEqual((bare.where, bare.time_ns), ("", 0))


class ReportedErrors(unittest.TestCase):
    """A failure raises Error, and its message names the problem."""

    def setUp(self):
        self.graph = _gr4.RuntimeGraph()
        self.source = self.graph.emplace(COUNTING_SOURCE, "source")
        self.sink = self.graph.emplace(NULL_SINK, "sink")

    def test_an_unknown_block_type_is_named(self):
        with self.assertRaises(_gr4.Error) as caught:
            self.graph.emplace("gr::blocks::testing::NoSuchBlock<float32>", "missing")
        self.assertIn("gr::blocks::testing::NoSuchBlock<float32>", caught.exception.message)
        self.assertEqual(caught.exception.where, "RuntimeGraph::emplace")

    def test_an_unknown_port_is_named(self):
        with self.assertRaises(_gr4.Error) as caught:
            self.graph.connect(self.source, "no_such_port", self.sink, "in")
        self.assertIn("no_such_port", caught.exception.message)
        self.graph.connect(self.source, "out", self.sink, "in")
        self.assertEqual(len(self.graph.edges()), 1)

    def test_malformed_yaml_names_the_line_and_column(self):
        malformed = 'blocks:\n  - id: gr::blocks::testing::NullSink<float32>\n    parameters:\n      name: "sink\n'
        with self.assertRaises(_gr4.Error) as caught:
            _gr4.RuntimeGraph.from_yaml(malformed)
        self.assertRegex(caught.exception.message, r"line \d+, column \d+")
        self.assertEqual(caught.exception.where, "RuntimeGraph::fromYaml")
        loaded = _gr4.RuntimeGraph.from_yaml(self.graph.to_yaml())
        self.assertEqual(len(loaded.blocks()), 2)

    def test_a_cpp_exception_a_call_lets_out_is_raised_as_error(self):
        nested = 1
        for _ in range(9):
            nested = [nested]
        with self.assertRaisesRegex(_gr4.Error, "rank"):
            _gr4.convert(nested, "tensor<int8>")
        self.assertEqual(_gr4.convert([[[1]]], "tensor<int8>"), [[[1]]])


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
