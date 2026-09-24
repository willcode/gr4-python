"""Tests of schedulers: runs, stops, interrupts, errors, events, requests, subscriptions and the exit of the process.

Every graph that writes a file bounds its samples with a head block and writes into a temporary directory; a graph
that runs until it is stopped ends in a null sink.
"""

import array
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest

import gr4

SIGNAL_GENERATOR = "gr::blocks::basic::SignalGenerator<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"
FILE_SINK = "gr::blocks::fileio::BasicFileSink<float32>"
FILE_SOURCE = "gr::blocks::fileio::BasicFileSource<float32>"
NULL_SOURCE = "gr::blocks::testing::NullSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
COPY = "gr::blocks::testing::Copy<float32>"
MULTI_THREADED = "gr::scheduler::Simple<multiThreaded>"
RATE = 20_000_000
# The longest wait for a state or an event, in seconds; a failing test ends after it.
PATIENCE = 10.0


def endless_tone():
    """Returns a graph of a tone into a null sink, which runs until it is stopped, and the tone block."""
    graph = gr4.Graph()
    tone = graph.add(SIGNAL_GENERATOR, "tone", sample_rate=1_000_000, tone_frequency=440.0)
    graph.connect(tone, graph.add(NULL_SINK, "sink"))
    return graph, tone


def head_chain(path, samples=1000):
    """Returns a graph of a tone, a head block of samples and a file sink writing to path."""
    graph = gr4.Graph()
    tone = graph.add(SIGNAL_GENERATOR, "tone", sample_rate=RATE)
    graph.connect(tone, graph.add(HEAD, "head", n_samples_max=samples), graph.add(FILE_SINK, "sink", file_name=path))
    return graph


def missing_file_run(path):
    """Returns a graph of a file source reading path, which does not exist, into a null sink."""
    graph = gr4.Graph()
    graph.connect(graph.add(FILE_SOURCE, "source", file_name=path), graph.add(NULL_SINK, "sink"))
    return graph


class Forwarded:
    """Forwards every attribute to a runtime, except those given as keywords."""

    def __init__(self, runtime, **replaced):
        self._runtime = runtime
        vars(self).update(replaced)

    def __getattr__(self, name):
        return getattr(self._runtime, name)


def endless_nulls():
    """Returns a graph of a null source into a null sink, which runs until it is stopped."""
    graph = gr4.Graph()
    graph.connect(graph.add(NULL_SOURCE, "source"), graph.add(NULL_SINK, "sink"))
    return graph


def wait_for_state(scheduler, state):
    """Waits until scheduler reaches state, at most PATIENCE seconds, and returns the state it saw last."""
    deadline = time.monotonic() + PATIENCE
    while scheduler.state != state and time.monotonic() < deadline:
        time.sleep(0.001)
    return scheduler.state


def wait_for_value(block, key, value):
    """Waits until the setting key of block holds value, at most PATIENCE seconds, and returns the value it saw last."""
    deadline = time.monotonic() + PATIENCE
    while block.settings[key] != value and time.monotonic() < deadline:
        time.sleep(0.001)
    return block.settings[key]


def dispatch_threads():
    """Returns the dispatch threads alive in this process."""
    return [thread for thread in threading.enumerate() if thread.name == "gr4-dispatch"]


class TemporaryFiles(unittest.TestCase):
    """A test case with a temporary directory that is removed after each test."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def path(self, name):
        """Returns the path of name in the temporary directory."""
        return os.path.join(self.directory, name)


class Runs(TemporaryFiles):
    """run() runs to the end, a stopped scheduler runs again, and a run ends by a stop, an interrupt or an error."""

    def test_a_stopped_scheduler_runs_its_blocks_a_second_time(self):
        first, second = self.path("first.f32"), self.path("second.f32")
        graph = gr4.Graph()
        tone = graph.add(SIGNAL_GENERATOR, "tone", sample_rate=RATE)
        head = graph.add(HEAD, "head", n_samples_max=1000)
        sink = graph.add(FILE_SINK, "sink", file_name=first)
        graph.connect(tone, head, sink)
        scheduler = gr4.Scheduler(graph, "gr::scheduler::Simple<multiThreaded>", timeout_ms=50)
        self.addCleanup(scheduler.close)
        self.assertEqual(scheduler.settings["timeout_ms"], 50)
        scheduler.run()
        self.assertEqual(scheduler.state, gr4.State.STOPPED)
        self.assertEqual(os.path.getsize(first), 4000)
        # The sink opens the file it holds when it starts and takes the staged name at its first work call.
        sink.settings["file_name"] = second
        graph.run()
        self.assertEqual(os.path.getsize(second), 4000)

    def test_a_stop_from_another_thread_ends_a_run(self):
        scheduler = gr4.Scheduler(endless_nulls())
        self.addCleanup(scheduler.close)
        seen = []

        def stop_once_running():
            seen.append(wait_for_state(scheduler, gr4.State.RUNNING))
            scheduler.stop()

        stopper = threading.Thread(target=stop_once_running)
        stopper.start()
        scheduler.run()
        stopper.join()
        self.assertEqual(seen, [gr4.State.RUNNING])
        self.assertEqual(scheduler.state, gr4.State.STOPPED)
        self.assertEqual(scheduler.errors, [])

    def test_sigint_during_run_stops_the_graph_and_raises_keyboard_interrupt(self):
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, previous)
        scheduler = gr4.Scheduler(endless_nulls())
        self.addCleanup(scheduler.close)
        main = threading.main_thread().ident
        seen = []

        def interrupt_once_running():
            seen.append(wait_for_state(scheduler, gr4.State.RUNNING))
            signal.pthread_kill(main, signal.SIGINT)

        interrupter = threading.Thread(target=interrupt_once_running)
        interrupter.start()
        with self.assertRaises(KeyboardInterrupt):
            scheduler.run()
        interrupter.join()
        self.assertEqual(seen, [gr4.State.RUNNING])
        self.assertEqual(scheduler.state, gr4.State.STOPPED)
        self.assertTrue(scheduler.wait(0.0))

    def test_an_error_inside_a_block_reaches_run_error(self):
        missing = self.path("never-written.f32")
        graph = gr4.Graph()
        graph.connect(graph.add(FILE_SOURCE, "source", file_name=missing), graph.add(NULL_SINK, "sink"))
        with self.assertRaises(gr4.RunError) as caught:
            graph.run()
        self.addCleanup(graph.scheduler.close)
        reported = [event for event in caught.exception.events if missing in event.text]
        self.assertEqual(len(reported), 1, caught.exception.events)
        self.assertTrue(reported[0].is_error)
        self.assertEqual(list(caught.exception.events), graph.scheduler.errors)
        self.assertEqual(graph.scheduler.state, gr4.State.ERROR)
        self.assertEqual(caught.exception.where, reported[0].source)
        self.assertTrue(str(caught.exception).startswith(f"{reported[0].source}: {reported[0].text}; "))

    def test_the_background_pattern_yields_the_error_event_of_a_failed_run_and_raises_run_error(self):
        missing = self.path("never-written.f32")
        scheduler = gr4.Scheduler(missing_file_run(missing))
        self.addCleanup(scheduler.close)
        seen = []
        with self.assertRaises(gr4.RunError) as caught:
            with scheduler.start():
                for event in scheduler.events(timeout=PATIENCE):
                    seen.append(event)
        self.assertEqual([event.is_error for event in seen], [True])
        self.assertIn(missing, seen[0].text)
        self.assertEqual(list(caught.exception.events), seen)
        self.assertEqual(scheduler.state, gr4.State.ERROR)
        # An exception of the with block propagates in place of the run's RunError.
        with self.assertRaises(LookupError):
            with scheduler.start():
                self.assertEqual(wait_for_state(scheduler, gr4.State.ERROR), gr4.State.ERROR)
                raise LookupError("raised in the with block")

    def test_a_stop_before_the_first_run_leaves_the_state_and_the_next_run_whole(self):
        for policy in (None, MULTI_THREADED):
            with self.subTest(policy=policy):
                path = self.path(f"{policy}.f32")
                scheduler = gr4.Scheduler(head_chain(path), policy)
                self.addCleanup(scheduler.close)
                before = scheduler.state
                scheduler.stop()
                self.assertEqual(scheduler.state, before)
                scheduler.run()
                self.assertEqual(os.path.getsize(path), 4000)

    def test_an_interrupt_before_the_run_starts_leaves_the_next_run_whole(self):
        def interrupted():
            raise KeyboardInterrupt

        path = self.path("interrupted.f32")
        scheduler = gr4.Scheduler(head_chain(path))
        self.addCleanup(scheduler.close)
        runtime = scheduler._runtime
        scheduler._runtime = Forwarded(runtime, run_in_thread=interrupted)
        with self.assertRaises(KeyboardInterrupt):
            scheduler.run()
        scheduler._runtime = runtime
        self.assertFalse(os.path.exists(path))
        scheduler.run()
        self.assertEqual(os.path.getsize(path), 4000)

    def test_pause_and_resume_move_the_running_graph_between_states(self):
        scheduler = gr4.Scheduler(endless_nulls(), "gr::scheduler::Simple<multiThreaded>")
        self.addCleanup(scheduler.close)
        with scheduler.start():
            self.assertEqual(wait_for_state(scheduler, gr4.State.RUNNING), gr4.State.RUNNING)
            scheduler.pause()
            self.assertEqual(wait_for_state(scheduler, gr4.State.PAUSED), gr4.State.PAUSED)
            scheduler.resume()
            self.assertEqual(wait_for_state(scheduler, gr4.State.RUNNING), gr4.State.RUNNING)
        self.assertEqual(scheduler.state, gr4.State.STOPPED)

    def test_the_scheduler_type_and_its_settings_are_checked_before_the_graph_moves(self):
        graph = endless_nulls()
        with self.assertRaises(gr4.UnknownBlockType):
            gr4.Scheduler(graph, "gr::scheduler::NoSuchScheduler<singleThreaded>")
        with self.assertRaisesRegex(KeyError, "'timeout_ms'"):
            gr4.Scheduler(graph, timeout_m=50)
        self.assertIsNone(graph.scheduler)
        scheduler = gr4.Scheduler(graph)
        self.addCleanup(scheduler.close)
        self.assertIs(graph.scheduler, scheduler)
        with self.assertRaises(gr4.SchedulerError):
            gr4.Scheduler(graph)

    def test_staging_a_scheduler_setting_raises_scheduler_error_naming_the_construction_settings(self):
        scheduler = gr4.Scheduler(endless_nulls(), MULTI_THREADED, timeout_ms=50)
        self.addCleanup(scheduler.close)
        stagings = {
            "item": lambda: scheduler.settings.__setitem__("timeout_ms", 77),
            "update": lambda: scheduler.settings.update(timeout_ms=77),
        }
        for name, stage in stagings.items():
            with self.subTest(staging=name):
                with self.assertRaisesRegex(gr4.SchedulerError, r"gr4\.Scheduler\(graph, type, \*\*settings\)"):
                    stage()
        self.assertEqual(scheduler.settings.staged, {})
        self.assertEqual(scheduler.settings["timeout_ms"], 50)


class RunningGraph(unittest.TestCase):
    """A graph a scheduler runs refuses structural edits and writes its document through the scheduler."""

    def test_a_structural_edit_while_running_raises_graph_busy_error(self):
        graph, tone = endless_tone()
        sink = next(block for block in graph if block.name == "sink")
        scheduler = gr4.Scheduler(graph)
        self.addCleanup(scheduler.close)
        edits = {
            "add": lambda: graph.add(NULL_SINK, "extra"),
            "add_subgraph": lambda: graph.add_subgraph("inner"),
            "connect": lambda: graph.connect(tone, sink),
            "disconnect": lambda: graph.disconnect(tone, sink),
            "remove": lambda: graph.remove(sink),
        }
        with scheduler.start():
            for name, edit in edits.items():
                with self.subTest(edit=name), self.assertRaises(gr4.GraphBusyError):
                    edit()
            self.assertEqual(len(graph), 2)
        extra = graph.add(NULL_SINK, "extra")
        graph.remove(extra)
        self.assertEqual(len(graph), 2)

    def test_the_document_of_a_running_graph_is_the_scheduler_s_reply_and_equals_the_document_after_it_stops(self):
        graph, _ = endless_tone()
        scheduler = gr4.Scheduler(graph)
        self.addCleanup(scheduler.close)
        asked = []
        request = scheduler.request

        def recorded(service, endpoint, *arguments, **keywords):
            asked.append((service, endpoint))
            return request(service, endpoint, *arguments, **keywords)

        scheduler.request = recorded
        with scheduler.start():
            self.assertEqual(wait_for_state(scheduler, gr4.State.RUNNING), gr4.State.RUNNING)
            running = graph.to_yaml()
            self.assertIn("SignalGenerator", running)
            self.assertEqual(asked, [(scheduler, "GraphGRC")])
        self.assertEqual(scheduler.state, gr4.State.STOPPED)
        self.assertEqual(running, graph.to_yaml())
        self.assertEqual(len(asked), 1)

    def test_an_export_or_unexport_while_running_raises_graph_busy_error(self):
        graph = gr4.Graph()
        inner = graph.add_subgraph("inner")
        copy = inner.interior.add(COPY, "copy")
        inner.interior.export(copy.inputs["in"], "sub_in")
        inner.interior.export(copy.outputs["out"], "sub_out")
        graph.connect(graph.add(NULL_SOURCE, "source"), inner, graph.add(NULL_SINK, "sink"))
        output = copy.outputs["out"]
        scheduler = gr4.Scheduler(graph)
        self.addCleanup(scheduler.close)
        interior = inner.interior
        edits = {"export": lambda: interior.export(output, "again"), "unexport": lambda: interior.unexport(output)}
        with scheduler.start():
            for name, edit in edits.items():
                with self.subTest(edit=name), self.assertRaises(gr4.GraphBusyError):
                    edit()
        self.assertEqual(list(inner.outputs), ["sub_out"])

    def test_after_close_the_graph_stays_readable_and_refuses_structural_edits_and_runs(self):
        graph, tone = endless_tone()
        sink = next(block for block in graph if block.name == "sink")
        gr4.Scheduler(graph).close()
        self.assertEqual(len(graph), 2)
        self.assertEqual(tone.settings["tone_frequency"], 440.0)
        self.assertIn("SignalGenerator", graph.to_yaml())
        edits = {
            "add": lambda: graph.add(NULL_SINK, "extra"),
            "connect": lambda: graph.connect(tone, sink),
            "remove": lambda: graph.remove(sink),
            "run": graph.run,
            "Scheduler": lambda: gr4.Scheduler(graph),
        }
        for name, edit in edits.items():
            with self.subTest(edit=name), self.assertRaisesRegex(gr4.SchedulerError, "closed|belongs to a scheduler"):
                edit()
        self.assertEqual(len(graph), 2)


class Messages(unittest.TestCase):
    """Requests are answered by their id, and subscriptions report staged changes."""

    def setUp(self):
        self.graph, self.tone = endless_tone()
        self.scheduler = gr4.Scheduler(self.graph)
        self.addCleanup(self.scheduler.close)

    def test_a_request_needs_a_running_graph_and_returns_the_reply(self):
        with self.assertRaises(gr4.SchedulerError):
            self.tone.request("Settings")
        with self.scheduler.start():
            settings = self.tone.request("Settings")
            self.assertEqual(settings["tone_frequency"], 440.0)
            self.assertEqual(settings["unique_name"], self.tone.unique_name)
            document = self.scheduler.request(self.scheduler, "GraphGRC")
            self.assertIn("SignalGenerator", document["value"])

    def test_an_error_reply_raises_error_and_a_missing_reply_raises_timeout(self):
        with self.scheduler.start():
            with self.assertRaisesRegex(gr4.Error, "does not implement") as caught:
                self.tone.request("Echo", command=gr4.Command.GET)
            self.assertEqual(caught.exception.where, self.tone.unique_name)
            started = time.monotonic()
            with self.assertRaises(TimeoutError) as timed_out:
                self.tone.request("NoSuchEndpoint", timeout=0.2)
            self.assertGreaterEqual(time.monotonic() - started, 0.2)
            self.assertIsInstance(timed_out.exception, gr4.Timeout)
            self.assertEqual(self.tone.request("Settings")["tone_frequency"], 440.0)

    def test_a_staged_change_arrives_in_the_next_settings_notification_on_the_dispatch_thread(self):
        received = []
        arrived = threading.Event()

        def record(event):
            received.append((event, threading.current_thread().name))
            arrived.set()

        self.assertEqual(dispatch_threads(), [])
        with self.scheduler.start():
            subscription = self.tone.subscribe("Settings", record)
            self.assertEqual(len(dispatch_threads()), 1)
            # The block handles messages in order: the reply to this request follows the subscription.
            self.tone.request("Settings")
            self.assertEqual(received, [])
            self.tone.settings["tone_frequency"] = 880
            self.assertTrue(arrived.wait(PATIENCE))
            event, thread = received[0]
            self.assertEqual(event.data["tone_frequency"], 880.0)
            self.assertEqual((event.command, event.request_id), (gr4.Command.NOTIFY, subscription.request_id))
            self.assertEqual(thread, "gr4-dispatch")
            subscription.cancel()
            self.assertFalse(subscription.active)
            self.assertEqual(dispatch_threads(), [])

    def test_a_reply_after_its_request_timed_out_is_dropped(self):
        with self.scheduler.start():
            self.tone.request("Settings")
            # With the drain held off, the request cannot see its reply before its timeout.
            self.scheduler._drain = lambda: None
            with self.assertRaises(gr4.Timeout):
                self.tone.request("Echo", timeout=0.05)
            del self.scheduler._drain
            # The block answers in order: the late error reply to Echo arrives before this reply.
            self.assertEqual(self.tone.request("Settings")["tone_frequency"], 440.0)
            self.assertEqual(self.scheduler._expired, {})
            self.assertEqual(self.scheduler.errors, [])
            self.assertEqual([event for event in self.scheduler.events() if event.endpoint == "Echo"], [])

    def test_a_settings_payload_to_a_block_converts_to_the_block_s_setting_types(self):
        with self.scheduler.start():
            staged = self.tone.request("StagedSettings", {"sample_rate": 20_000_000}, gr4.Command.SET)
            self.assertEqual(staged["sample_rate"], 20_000_000.0)
            self.assertEqual(wait_for_value(self.tone, "sample_rate", 20_000_000.0), 20_000_000.0)
            # Addressed by its unique name, the payload keeps the Int64 its value selects, which the block refuses.
            with self.assertRaisesRegex(gr4.Error, "sample_rate"):
                self.scheduler.request(
                    self.tone.unique_name, "StagedSettings", {"sample_rate": 20_000_000}, gr4.Command.SET
                )
            self.scheduler.send(self.tone, "Settings", {"sample_rate": 16_777_217})
            self.assertEqual(wait_for_value(self.tone, "sample_rate", 16_777_216.0), 16_777_216.0)

    def test_a_callback_subscription_made_while_the_last_one_is_canceled_keeps_the_dispatch_thread(self):
        arrived = threading.Event()
        added = []
        send = self.scheduler._send

        def subscribe_during_unsubscribe(command, *arguments):
            send(command, *arguments)
            if command == gr4.Command.UNSUBSCRIBE and not added:
                added.append(self.tone.subscribe("Settings", lambda event: arrived.set()))

        with self.scheduler.start():
            first = self.tone.subscribe("Settings", lambda event: None)
            self.scheduler._send = subscribe_during_unsubscribe
            first.cancel()
            del self.scheduler._send
            self.assertEqual(len(added), 1)
            self.assertEqual(len(dispatch_threads()), 1)
            self.tone.request("Settings")
            self.tone.settings["tone_frequency"] = 770
            self.assertTrue(arrived.wait(PATIENCE))
            added[0].cancel()
            self.assertEqual(dispatch_threads(), [])

    def test_the_notifications_of_a_subscription_without_callback_reach_events(self):
        with self.scheduler.start():
            with self.tone.subscribe("Settings") as subscription:
                self.tone.request("Settings")
                self.tone.settings.update(tone_frequency=660)
                notified = next(
                    (event for event in self.scheduler.events(PATIENCE) if event.request_id == subscription.request_id),
                    None,
                )
            self.assertFalse(subscription.active)
        self.assertIsNotNone(notified)
        self.assertEqual(notified.data["tone_frequency"], 660.0)
        self.assertEqual(dispatch_threads(), [])


class Delivery(unittest.TestCase):
    """A drain routes a whole batch, and close() waits a bounded time for a callback."""

    def test_an_interrupt_while_a_batch_is_routed_leaves_the_rest_of_the_batch_routed(self):
        scheduler = gr4.Scheduler(endless_nulls())
        self.addCleanup(scheduler.close)
        batch = [
            types.SimpleNamespace(
                source="source",
                endpoint=f"endpoint-{index}",
                command=gr4.Command.NOTIFY,
                client_request_id="",
                is_error=False,
                text="",
                data={"index": index},
                time_ns=0,
            )
            for index in range(3)
        ]
        batches = [batch]
        runtime = scheduler._runtime
        scheduler._runtime = Forwarded(runtime, poll_events=lambda count: batches.pop() if batches else [])
        self.addCleanup(setattr, scheduler, "_runtime", runtime)
        route = scheduler._route
        routed = []

        def interrupted(event):
            routed.append(event.endpoint)
            if len(routed) == 1:
                raise KeyboardInterrupt
            route(event)

        scheduler._route = interrupted
        with self.assertRaises(KeyboardInterrupt):
            scheduler._drain()
        del scheduler._route
        self.assertEqual(routed, ["endpoint-0", "endpoint-1", "endpoint-2"])
        self.assertEqual([event.endpoint for event in scheduler.events()], ["endpoint-1", "endpoint-2"])

    def test_close_waits_close_timeout_for_a_callback_and_leaves_it_with_a_warning(self):
        graph, tone = endless_tone()
        scheduler = gr4.Scheduler(graph)
        self.addCleanup(scheduler.close)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def blocked(event):
            entered.set()
            release.wait(PATIENCE)

        scheduler.start()
        tone.subscribe("Settings", blocked)
        tone.request("Settings")
        tone.settings["tone_frequency"] = 880
        self.assertTrue(entered.wait(PATIENCE))
        (thread,) = dispatch_threads()
        started = time.monotonic()
        with self.assertWarnsRegex(RuntimeWarning, "still in a callback"):
            scheduler.close()
        self.assertGreaterEqual(time.monotonic() - started, gr4.scheduler.CLOSE_TIMEOUT)
        self.assertTrue(thread.is_alive())
        release.set()
        thread.join(PATIENCE)
        self.assertFalse(thread.is_alive())


class Exit(unittest.TestCase):
    """An interpreter that exits with a graph running closes the scheduler and exits cleanly."""

    def test_an_interpreter_exits_cleanly_with_a_graph_running(self):
        program = textwrap.dedent(f"""
            import gr4
            graph = gr4.Graph()
            tone = graph.add({SIGNAL_GENERATOR!r}, "tone", sample_rate=1_000_000)
            graph.connect(tone, graph.add({NULL_SINK!r}, "sink"))
            scheduler = gr4.Scheduler(graph, "gr::scheduler::Simple<multiThreaded>").start()
            tone.subscribe("Settings", lambda event: None)
            tone.request("Settings")
            print(scheduler.state.name, flush=True)
            """)
        finished = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60, env=dict(os.environ)
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(finished.stdout.split(), ["RUNNING"])
        self.assertEqual(finished.stderr, "")

    def test_an_interpreter_exits_with_a_callback_that_never_returns_and_warns(self):
        program = textwrap.dedent(f"""
            import threading
            import gr4
            graph = gr4.Graph()
            tone = graph.add({SIGNAL_GENERATOR!r}, "tone", sample_rate=1_000_000)
            graph.connect(tone, graph.add({NULL_SINK!r}, "sink"))
            scheduler = gr4.Scheduler(graph).start()
            entered = threading.Event()

            def blocked(event):
                entered.set()
                threading.Event().wait()

            tone.subscribe("Settings", blocked)
            tone.request("Settings")
            tone.settings["tone_frequency"] = 880.0
            print(entered.wait(10.0), flush=True)
            """)
        finished = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=20, env=dict(os.environ)
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(finished.stdout.split(), ["True"])
        self.assertIn("RuntimeWarning", finished.stderr)
        self.assertIn("still in a callback", finished.stderr)


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
