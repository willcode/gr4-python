"""Tests of the message plane: requests, their replies, and the scheduler's own settings."""

import enum
import os
import sys
import time
import unittest

from gr4 import _gr4

COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
SAMPLES = 1 << 20


def counted_chain():
    """Returns a graph of a counting source, a head block of SAMPLES and a null sink, and the head block."""
    graph = _gr4.RuntimeGraph()
    source = graph.emplace(COUNTING_SOURCE, "source")
    head = graph.emplace(HEAD, "head", {"n_samples_max": SAMPLES}, {"n_samples_max": "uint64"})
    sink = graph.emplace(NULL_SINK, "sink")
    graph.connect(source, "out", head, "in")
    graph.connect(head, "out", sink, "in")
    return graph, head


class Requests(unittest.TestCase):
    """A request sent to a block is answered on the message plane."""

    def test_a_get_with_a_request_id_is_answered_by_a_final_event_with_that_id(self):
        graph, head = counted_chain()
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)
        runtime.send(_gr4.Command.GET, head.unique_name, "Settings", None, "get-7")
        runtime.run_and_wait()
        events = runtime.poll_events(1024)
        replies = [
            event
            for event in events
            if event.source == head.unique_name
            and event.endpoint == "Settings"
            and event.command == _gr4.Command.FINAL
            and event.client_request_id == "get-7"
        ]
        self.assertEqual(len(replies), 1, events)
        self.assertFalse(replies[0].is_error, replies[0].text)
        self.assertEqual(replies[0].data["n_samples_max"], SAMPLES)

    def test_a_set_of_a_value_of_the_wrong_type_is_answered_by_an_error_event(self):
        graph, head = counted_chain()
        runtime = _gr4.Runtime.create(graph)
        self.addCleanup(runtime.close)
        sent = time.time_ns()
        runtime.send(_gr4.Command.SET, head.unique_name, "Settings", {"n_samples_max": "many"}, "set-9")
        runtime.run_and_wait()
        ended = time.time_ns()
        replies = [event for event in runtime.poll_events(1024) if event.client_request_id == "set-9"]
        self.assertEqual(len(replies), 1, replies)
        (reply,) = replies
        self.assertTrue(reply.is_error)
        self.assertEqual((reply.source, reply.command), (head.unique_name, _gr4.Command.FINAL))
        self.assertIn("n_samples_max", reply.text)
        self.assertTrue(sent <= reply.time_ns <= ended, (sent, reply.time_ns, ended))
        self.assertEqual(head.get_value("n_samples_max"), SAMPLES)

    def test_the_scheduler_reports_the_timeout_given_to_create(self):
        defaults = _gr4.Runtime.create(counted_chain()[0])
        self.addCleanup(defaults.close)
        chosen = defaults.scheduler().get_value("timeout_ms") + 23
        runtime = _gr4.Runtime.create(counted_chain()[0], None, {"timeout_ms": chosen}, {"timeout_ms": "uint32"})
        self.addCleanup(runtime.close)
        scheduler = runtime.scheduler()
        self.assertEqual(scheduler.get_value("timeout_ms"), chosen)
        self.assertEqual(scheduler.setting_types()["timeout_ms"], "uint32")


class Enumerations(unittest.TestCase):
    """State and Command are IntEnums with the values of the C++ enumerations."""

    def test_the_enumerations_carry_the_cpp_values(self):
        self.assertTrue(issubclass(_gr4.State, enum.IntEnum))
        self.assertTrue(issubclass(_gr4.Command, enum.IntEnum))
        states = ["IDLE", "INITIALIZED", "RUNNING", "REQUESTED_PAUSE", "PAUSED", "REQUESTED_STOP", "STOPPED", "ERROR"]
        self.assertEqual([state.name for state in _gr4.State], states)
        self.assertEqual([state.value for state in _gr4.State], list(range(len(states))))
        commands = ["INVALID", "GET", "SET", "PARTIAL", "FINAL", "READY", "DISCONNECT", "SUBSCRIBE", "UNSUBSCRIBE"]
        commands += ["NOTIFY", "HEARTBEAT"]
        self.assertEqual([command.name for command in _gr4.Command], commands)
        self.assertEqual([command.value for command in _gr4.Command], list(range(len(commands))))


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
