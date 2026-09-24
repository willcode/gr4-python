"""Tests of the package's exceptions and of the translation of the extension's errors into them."""

import os
import sys
import unittest

import gr4
from gr4 import _gr4, errors

NULL_SOURCE = "gr::blocks::testing::NullSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"

BUILTIN_BASES = {
    gr4.UnknownBlockType: LookupError,
    gr4.PortError: LookupError,
    gr4.SettingError: ValueError,
    gr4.GraphFileError: ValueError,
    gr4.GraphBusyError: None,
    gr4.SchedulerError: None,
    gr4.RunError: None,
    gr4.Timeout: TimeoutError,
}


class Hierarchy(unittest.TestCase):
    """Every package exception is a gr4.Error, the extension's class, and the built-in exception of its kind."""

    def test_gr4_error_is_the_extension_error(self):
        self.assertIs(gr4.Error, _gr4.Error)

    def test_each_error_is_a_gr4_error_and_its_builtin_kind(self):
        for kind, builtin in BUILTIN_BASES.items():
            with self.subTest(kind=kind.__name__):
                error = kind("reason", "Where::method", 7)
                self.assertIsInstance(error, gr4.Error)
                self.assertEqual((error.message, error.where, error.time_ns), ("reason", "Where::method", 7))
                self.assertEqual(str(error), "reason")
                if builtin is not None:
                    self.assertIsInstance(error, builtin)

    def test_a_timeout_is_caught_as_timeout_error(self):
        with self.assertRaises(TimeoutError) as caught:
            raise gr4.Timeout("no reply", "Scheduler.request")
        self.assertEqual(caught.exception.where, "Scheduler.request")


class Translation(unittest.TestCase):
    """translated() raises an extension error as the kind the operation names, and passes everything else."""

    def test_an_extension_error_becomes_the_named_kind_with_its_fields(self):
        graph = _gr4.RuntimeGraph()
        source = graph.emplace(NULL_SOURCE, "source")
        sink = graph.emplace(NULL_SINK, "sink")
        with self.assertRaises(_gr4.Error) as direct:
            graph.connect(source, "no_such_port", sink, "in")
        with self.assertRaises(gr4.PortError) as caught:
            with errors.translated(gr4.PortError):
                graph.connect(source, "no_such_port", sink, "in")
        self.assertEqual(caught.exception.message, direct.exception.message)
        self.assertEqual(caught.exception.where, direct.exception.where)
        self.assertIn("no_such_port", caught.exception.message)

    def test_a_package_error_and_a_foreign_exception_pass_unchanged(self):
        with self.assertRaises(gr4.SettingError):
            with errors.translated(gr4.PortError):
                raise gr4.SettingError("refused")
        with self.assertRaises(KeyError):
            with errors.translated(gr4.PortError):
                raise KeyError("key")


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
