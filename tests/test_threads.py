"""Tests of runs on the extension's thread, of stops and closes from other Python threads, of Ctrl-C, and of the GIL
during a run."""

import os
import signal
import subprocess
import sys
import threading
import time
import unittest

from gr4 import _gr4

NULL_SOURCE = "gr::blocks::testing::NullSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
COUNT = 100_000
REPETITIONS = 20
# The longest a stop or a close may take to end a run before a case fails.
JOIN_TIMEOUT = 30.0


def endless_graph():
    """Returns a graph that runs until it is stopped: a null source into a null sink."""
    graph = _gr4.RuntimeGraph()
    source = graph.emplace(NULL_SOURCE, "source")
    sink = graph.emplace(NULL_SINK, "sink")
    graph.connect(source, "out", sink, "in")
    return graph


def until_running(runtime):
    """Waits until the scheduler reports RUNNING, at most JOIN_TIMEOUT seconds, and returns the state it reports."""
    deadline = time.monotonic() + JOIN_TIMEOUT
    while runtime.state != _gr4.State.RUNNING and time.monotonic() < deadline:
        time.sleep(0.001)
    return runtime.state


class Worker(threading.Thread):
    """A daemon thread that starts at once and keeps what its call returned or raised."""

    def __init__(self, call):
        super().__init__(daemon=True)
        self.call = call
        self.returned = None
        self.raised = None
        self.start()

    def run(self):
        try:
            self.returned = self.call()
        except BaseException as error:
            self.raised = error


def finish(test, *workers):
    """Joins the workers, JOIN_TIMEOUT seconds each at most, and fails test when one is alive or raised."""
    for worker in workers:
        worker.join(JOIN_TIMEOUT)
    test.assertEqual([worker.is_alive() for worker in workers], [False] * len(workers))
    test.assertEqual([worker.raised for worker in workers], [None] * len(workers))


class RunInThread(unittest.TestCase):
    """run_in_thread() runs on a thread the extension owns, and wait() blocks without the GIL."""

    def setUp(self):
        self.runtime = _gr4.Runtime.create(endless_graph())
        self.addCleanup(self.runtime.close)

    def test_a_stop_from_a_python_thread_ends_the_run_another_thread_waits_for(self):
        self.runtime.run_in_thread()
        self.assertFalse(self.runtime.wait(0.05))
        with self.assertRaises(_gr4.Error):
            self.runtime.run_in_thread()
        with self.assertRaises(_gr4.Error):
            self.runtime.result()

        # The counter starts once this thread is about to wait, and only its stop() ends the run: a wait()
        # holding the GIL would keep the counter at zero and time out.
        waiting = threading.Event()
        count = [0]

        def count_then_stop():
            waiting.wait()
            while count[0] < COUNT:
                count[0] += 1
            self.runtime.stop()

        counter = threading.Thread(target=count_then_stop)
        counter.start()
        waiting.set()
        ended = self.runtime.wait(timeout=60.0)
        counter.join()
        self.assertTrue(ended)
        self.assertEqual(count[0], COUNT)
        self.assertIsNone(self.runtime.result())
        self.assertEqual(self.runtime.state, _gr4.State.STOPPED)

    def test_close_stops_and_joins_a_running_thread(self):
        self.runtime.run_in_thread()
        self.assertFalse(self.runtime.wait(0.05))
        self.runtime.close()
        self.assertTrue(self.runtime.wait(0.0))
        with self.assertRaisesRegex(_gr4.Error, "closed"):
            self.runtime.state

    def test_wait_refuses_a_nan_timeout(self):
        self.runtime.run_in_thread()
        with self.assertRaises(ValueError):
            self.runtime.wait(float("nan"))
        self.assertFalse(self.runtime.wait(0.0))

    def test_a_quiesce_block_holds_its_guard_from_entry_to_exit(self):
        self.runtime.run_in_thread()
        self.assertEqual(until_running(self.runtime), _gr4.State.RUNNING)
        with self.runtime.quiesce() as quiesced:
            with self.assertRaisesRegex(_gr4.Error, "entered already"):
                quiesced.__enter__()
            with self.assertRaisesRegex(_gr4.Error, "inside a quiesce"):
                self.runtime.close()
            self.assertEqual(self.runtime.state, _gr4.State.RUNNING)
        self.runtime.close()
        self.assertTrue(self.runtime.wait(0.0))

    def test_a_thread_that_cannot_start_raises_error_and_the_next_run_starts(self):
        # A thread limit of one makes every thread start fail. The first run ends by itself and its thread is left
        # unjoined, so the refused start has an ended thread to join.
        program = (
            "import os, resource, sys, time\n"
            "from gr4 import _gr4\n"
            "graph = _gr4.RuntimeGraph()\n"
            "source = graph.emplace('gr::blocks::testing::CountingSource<float32>', 'source')\n"
            "head = graph.emplace('gr::blocks::testing::HeadBlock<float32>', 'head', {'n_samples_max': 100},"
            " {'n_samples_max': 'uint32'})\n"
            "sink = graph.emplace('gr::blocks::testing::NullSink<float32>', 'sink')\n"
            "graph.connect(source, 'out', head, 'in')\n"
            "graph.connect(head, 'out', sink, 'in')\n"
            "runtime = _gr4.Runtime.create(graph)\n"
            "runtime.run_in_thread()\n"
            "deadline = time.monotonic() + 30.0\n"
            "while time.monotonic() < deadline:\n"
            "    try:\n"
            "        runtime.result()\n"
            "        break\n"
            "    except _gr4.Error:\n"
            "        time.sleep(0.001)\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))\n"
            "try:\n"
            "    runtime.run_in_thread()\n"
            "    print('started')\n"
            "except _gr4.Error as error:\n"
            "    print('refused', error.where, error.message)\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))\n"
            "runtime.run_in_thread()\n"
            "print('next run ended', runtime.wait(30.0))\n"
            "runtime.close()\n"
            "sys.stdout.flush()\n"
            "os._exit(0)\n"
        )
        run = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        refused, ended = run.stdout.splitlines()
        self.assertTrue(refused.startswith("refused Runtime.run_in_thread the run's thread did not start"), refused)
        self.assertEqual(ended, "next run ended True")


class OneLifecycle(unittest.TestCase):
    """A runtime takes one run at a time, and close() from any thread ends it, however the calls interleave."""

    def test_a_second_run_is_refused_while_run_and_wait_runs(self):
        runtime = _gr4.Runtime.create(endless_graph())
        self.addCleanup(runtime.close)
        runner = Worker(runtime.run_and_wait)
        self.assertEqual(until_running(runtime), _gr4.State.RUNNING)
        for start in (runtime.run_in_thread, runtime.start, runtime.run_and_wait):
            with self.subTest(call=start.__name__), self.assertRaisesRegex(_gr4.Error, "in progress"):
                start()
        runtime.stop()
        finish(self, runner)
        self.assertEqual(runtime.state, _gr4.State.STOPPED)

    def test_two_threads_closing_one_runtime_both_return(self):
        for repetition in range(REPETITIONS):
            with self.subTest(repetition=repetition):
                runtime = _gr4.Runtime.create(endless_graph())
                runtime.run_in_thread()
                self.assertEqual(until_running(runtime), _gr4.State.RUNNING)
                barrier = threading.Barrier(2)
                finish(self, *[Worker(lambda: (barrier.wait(), runtime.close())) for _ in range(2)])
                self.assertTrue(runtime.wait(0.0))
                with self.assertRaisesRegex(_gr4.Error, "closed"):
                    runtime.state

    def test_close_from_another_thread_ends_a_run_and_wait(self):
        for repetition in range(REPETITIONS):
            with self.subTest(repetition=repetition):
                runtime = _gr4.Runtime.create(endless_graph())
                runner = Worker(runtime.run_and_wait)
                self.assertEqual(until_running(runtime), _gr4.State.RUNNING)
                finish(self, Worker(runtime.close), runner)
                with self.assertRaisesRegex(_gr4.Error, "closed"):
                    runtime.state

    def test_close_from_another_thread_ends_an_untimed_wait(self):
        for repetition in range(REPETITIONS):
            with self.subTest(repetition=repetition):
                runtime = _gr4.Runtime.create(endless_graph())
                runtime.run_in_thread()
                self.assertEqual(until_running(runtime), _gr4.State.RUNNING)
                waiter = Worker(runtime.wait)
                finish(self, Worker(runtime.close), waiter)
                self.assertIs(waiter.returned, True)


class RunAndWait(unittest.TestCase):
    """run_and_wait() blocks without the GIL."""

    def test_a_python_thread_runs_while_another_is_inside_run_and_wait(self):
        runtime = _gr4.Runtime.create(endless_graph())
        self.addCleanup(runtime.close)
        observed = []

        # RUNNING holds only inside run_and_wait(), and only this thread's stop() ends the run.
        def stop_once_running():
            deadline = time.monotonic() + 60.0
            while runtime.state != _gr4.State.RUNNING and time.monotonic() < deadline:
                time.sleep(0.001)
            observed.append(runtime.state)
            runtime.stop()

        stopper = threading.Thread(target=stop_once_running)
        stopper.start()
        runtime.run_and_wait()
        stopper.join()
        self.assertEqual(observed, [_gr4.State.RUNNING])
        self.assertEqual(runtime.state, _gr4.State.STOPPED)


class Interrupts(unittest.TestCase):
    """SIGINT, sent by another thread once the graph runs, raises KeyboardInterrupt in the call that waits for it."""

    def setUp(self):
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, previous)
        self.runtime = _gr4.Runtime.create(endless_graph())
        self.addCleanup(self.runtime.close)
        # Only this stop, JOIN_TIMEOUT seconds on, ends the run when the interrupt does not reach the call.
        self.fallback = threading.Timer(JOIN_TIMEOUT, self.runtime.stop)
        self.fallback.start()
        self.addCleanup(self.fallback.cancel)

    def interrupt_once_running(self):
        """Returns a thread that sends SIGINT to this process once the scheduler is RUNNING."""
        return Worker(lambda: (until_running(self.runtime), os.kill(os.getpid(), signal.SIGINT)))

    def test_ctrl_c_stops_and_joins_a_run_and_wait_and_raises_keyboard_interrupt(self):
        sender = self.interrupt_once_running()
        with self.assertRaises(KeyboardInterrupt):
            self.runtime.run_and_wait()
        self.assertTrue(self.fallback.is_alive())
        finish(self, sender)
        self.assertTrue(self.runtime.wait(0.0))
        self.assertEqual(self.runtime.state, _gr4.State.STOPPED)

    def test_ctrl_c_raises_keyboard_interrupt_from_an_untimed_wait_and_leaves_the_run_going(self):
        self.runtime.run_in_thread()
        sender = self.interrupt_once_running()
        with self.assertRaises(KeyboardInterrupt):
            self.runtime.wait()
        self.assertTrue(self.fallback.is_alive())
        finish(self, sender)
        self.assertFalse(self.runtime.wait(0.0))
        self.assertEqual(self.runtime.state, _gr4.State.RUNNING)
        self.runtime.stop()
        self.assertTrue(self.runtime.wait(0.0))


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
