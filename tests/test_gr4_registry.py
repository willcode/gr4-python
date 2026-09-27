"""Tests of the registry: listing, finding and describing block types, and choosing the plugin directories."""

import os
import pydoc
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import gr4
from gr4 import _gr4, registry

SIGNAL_GENERATOR = "gr::blocks::basic::SignalGenerator<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"


def missing_directory(case):
    """Returns the path of a directory that does not exist, inside a temporary directory removed after the test."""
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return os.path.join(directory.name, "absent")


def run_python(program, environment):
    """Runs program in a fresh interpreter with environment and returns its completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


class Listing(unittest.TestCase):
    """The registry lists, searches and finds the keys of the types this process can build."""

    def test_the_lists_hold_the_plugin_blocks_and_the_default_scheduler(self):
        self.assertIn(HEAD, registry.block_types())
        self.assertIn(gr4.DEFAULT_SCHEDULER, registry.scheduler_types())
        self.assertEqual(len(registry), len(registry.block_types()) + len(registry.scheduler_types()))
        self.assertIn(HEAD, registry)
        self.assertNotIn("gr::blocks::testing::HeadBlock<float128>", registry)

    def test_search_ignores_case_and_returns_only_matching_keys(self):
        found = registry.search("headblock")
        self.assertIn(HEAD, found)
        self.assertEqual(found, registry.search("HeadBlock"))
        self.assertTrue(all("headblock" in key.lower() for key in found))

    def test_find_returns_the_one_key_its_type_parameters_select(self):
        self.assertEqual(registry.find("HeadBlock", "float32"), HEAD)
        self.assertEqual(registry.find("gr::blocks::testing::HeadBlock", "float32"), HEAD)
        self.assertEqual(
            registry.find("HeadBlock", "complex<float32>"), "gr::blocks::testing::HeadBlock<complex<float32>>"
        )
        self.assertEqual(registry.find("Simple", "multiThreaded"), "gr::scheduler::Simple<multiThreaded>")

    def test_find_names_the_types_that_exist_when_it_selects_none_or_several(self):
        with self.assertRaises(gr4.UnknownBlockType) as none:
            registry.find("HeadBlock", "float128")
        self.assertIn("<float32>", none.exception.message)
        self.assertIsInstance(none.exception, LookupError)
        with self.assertRaisesRegex(gr4.UnknownBlockType, "<int16>"):
            registry.find("HeadBlock")
        with self.assertRaisesRegex(gr4.UnknownBlockType, "'HeadBlock'"):
            registry.find("HeadBlok", "float32")

    def test_find_names_the_namespaces_when_two_hold_the_family(self):
        keys = (
            "acme::HeadBlock<float32>",
            "gr::blocks::testing::HeadBlock<float32>",
            "gr::blocks::testing::HeadBlock<int16>",
            "other::testing::HeadBlock<float32>",
        )
        with mock.patch.object(registry, "_every_key", return_value=keys):
            for type_parameters in (("float32",), ()):
                with self.assertRaises(gr4.UnknownBlockType) as both:
                    registry.find("HeadBlock", *type_parameters)
                message = both.exception.message
                for family in ("'acme::HeadBlock'", "'gr::blocks::testing::HeadBlock'", "'other::testing::HeadBlock'"):
                    self.assertIn(family, message)
                self.assertIn("namespace component selects one, as in 'acme::HeadBlock'", message)
                self.assertNotIn("type parameters select one", message)
            with self.assertRaises(gr4.UnknownBlockType) as deeper:
                registry.find("testing::HeadBlock", "float32")
            self.assertIn("as in 'blocks::testing::HeadBlock'", deeper.exception.message)
            self.assertEqual(registry.find("acme::HeadBlock"), keys[0])
            self.assertEqual(registry.find("blocks::testing::HeadBlock", "int16"), keys[2])
            with self.assertRaisesRegex(gr4.UnknownBlockType, "type parameters select one"):
                registry.find("gr::blocks::testing::HeadBlock")

    def test_an_unknown_key_is_refused_with_the_closest_keys(self):
        with self.assertRaisesRegex(gr4.UnknownBlockType, "HeadBlock<float32>"):
            registry["gr::blocks::testing::HeadBlok<float32>"]


class Description(unittest.TestCase):
    """A BlockType describes the settings and ports of a default-constructed instance, read once per process."""

    def test_the_description_is_what_a_default_constructed_block_reports(self):
        graph = _gr4.RuntimeGraph()
        block = graph.emplace(SIGNAL_GENERATOR, "live")
        described = registry[SIGNAL_GENERATOR]
        types = block.setting_types()
        self.assertEqual(
            {name: setting.type for name, setting in described.settings.items()},
            {name: code for name, code in types.items() if name not in ("name", "unique_name")},
        )
        frequency = described.settings["tone_frequency"]
        self.assertEqual(frequency.default, block.get_value("tone_frequency"))
        self.assertEqual(frequency.unit, block.meta_information()["tone_frequency::unit"])
        self.assertEqual(frequency.unit, "Hz")
        signal_type = described.settings["signal_type"]
        self.assertIn(signal_type.default, signal_type.choices)
        self.assertIsNone(frequency.choices)
        self.assertEqual([port.name for port in described.outputs], graph.output_port_names(block))
        self.assertEqual([port.name for port in described.inputs], graph.input_port_names(block))
        self.assertEqual(described.outputs[0].type, "float32")
        self.assertEqual(described.type_parameters, ("float32",))

    def test_help_shows_the_description_of_the_type(self):
        text = pydoc.render_doc(registry[HEAD], renderer=pydoc.plaintext)
        self.assertIn("n_samples_max (uint32", text)
        self.assertIn("Limits the number of output samples", text)
        self.assertIn("A block or scheduler type", pydoc.render_doc(registry.BlockType, renderer=pydoc.plaintext))

    def test_the_description_is_read_once(self):
        key = "gr::blocks::testing::Copy<float32>"
        block_type = registry[key]
        self.assertIs(registry[key], block_type)
        self.assertFalse(block_type.described)
        graph = _gr4.RuntimeGraph()
        before = graph.emplace(key, "before").unique_name
        block_type.settings
        between = graph.emplace(key, "between").unique_name
        block_type.settings
        block_type.inputs
        after = graph.emplace(key, "after").unique_name

        def counter(unique_name):
            return int(unique_name.rsplit("#", 1)[1])

        self.assertEqual(counter(between) - counter(before), 2)
        self.assertEqual(counter(after) - counter(between), 1)


class DescribeAll(unittest.TestCase):
    """describe_all() reads descriptions in child processes and survives one that dies."""

    def test_the_children_describe_what_the_process_would(self):
        keys = ["gr::blocks::testing::NullSink<int16>", "gr::blocks::testing::NullSource<int16>"]
        described = registry.describe_all(keys + ["gr::blocks::testing::NoSuchBlock<int16>"], per_process=1)
        self.assertEqual(set(described.types), set(keys))
        self.assertEqual(set(described.failures), {"gr::blocks::testing::NoSuchBlock<int16>"})
        graph = _gr4.RuntimeGraph()
        for key in keys:
            with self.subTest(key=key):
                block = graph.emplace(key, "live")
                self.assertEqual(
                    {name: setting.type for name, setting in described.types[key].settings.items()},
                    {name: code for name, code in block.setting_types().items() if name not in ("name", "unique_name")},
                )
                self.assertIs(described.types[key], registry[key])

    def test_a_child_that_dies_costs_its_key_and_the_next_child_resumes_after_it(self):
        keys = [f"gr::blocks::testing::{family}<int32>" for family in ("Copy", "HeadBlock", "NullSink", "NullSource")]
        doomed = "gr::blocks::testing::CountingSource<int32>"
        # A child program that kills its own process when it reaches the doomed key.
        dying = textwrap.dedent(f"""
            import os, signal, sys
            from gr4 import registry
            read = registry._read_description
            registry._read_description = lambda key: os.kill(os.getpid(), signal.SIGKILL) if key == {doomed!r} else read(key)
            registry._serve_descriptions(int(sys.argv[1]))
            """)
        self.addCleanup(setattr, registry, "_CHILD_PROGRAM", registry._CHILD_PROGRAM)
        registry._CHILD_PROGRAM = dying
        described = registry.describe_all(keys[:2] + [doomed] + keys[2:], per_process=3)
        self.assertEqual(set(described.types), set(keys))
        self.assertEqual(list(described.failures), [doomed])
        self.assertIn("SIGKILL", described.failures[doomed])
        self.assertFalse(registry[doomed].described)

    def test_the_children_read_the_plugin_directories_the_loader_of_this_process_read(self):
        key = registry.find("Copy", "int16")
        self.assertFalse(registry[key].described)
        with mock.patch.dict(os.environ, GNURADIO4_PLUGIN_DIRECTORIES=missing_directory(self)):
            described = registry.describe_all([key], per_process=1)
        self.assertEqual(described.failures, {})
        self.assertEqual(list(described.types), [key])

    def test_an_interrupt_while_a_child_describes_kills_and_reaps_the_child(self):
        key = registry.find("HeadBlock", "int16")
        self.assertFalse(registry[key].described)
        children = []
        start = subprocess.Popen

        def recorded(*args, **kwargs):
            children.append(start(*args, **kwargs))
            return children[-1]

        with (
            mock.patch.object(registry.subprocess, "Popen", recorded),
            mock.patch.object(registry.select, "select", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            registry.describe_all([key], per_process=1)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        self.assertFalse(registry[key].described)

    def test_without_keys_every_block_and_scheduler_key_is_described(self):
        asked = []

        def describe_none(keys, timeout):
            asked.extend(keys)
            return [(key, None, "not described") for key in keys], None, ""

        with mock.patch.object(registry, "_describe_in_child", describe_none):
            described = registry.describe_all()
        self.assertEqual(set(described.types) | set(described.failures), set(registry))
        self.assertLessEqual(set(registry.scheduler_types()), set(described.types) | set(described.failures))
        self.assertEqual(set(asked), set(described.failures))


class Configure(unittest.TestCase):
    """configure() chooses the plugin directories before the loader's first use and refuses after it."""

    def test_configure_sets_the_directories_the_loader_reads(self):
        plugins = os.environ["GNURADIO4_PLUGIN_DIRECTORIES"]
        environment = dict(os.environ, GNURADIO4_PLUGIN_DIRECTORIES=missing_directory(self))
        program = """
            import os, sys
            import gr4
            {configure}
            print(sum("::testing::" in key for key in gr4.registry.block_types()))
            try:
                gr4.configure(plugin_dirs=[{plugins!r}])
            except gr4.Error:
                print("refused")
            sys.stdout.flush()
            os._exit(0)
            """
        unset = run_python(program.format(configure="", plugins=plugins), environment)
        configured = run_python(
            program.format(configure=f"gr4.configure(plugin_dirs=[{plugins!r}])", plugins=plugins), environment
        )
        self.assertEqual(unset.returncode, 0, unset.stderr)
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.assertEqual(unset.stdout.split(), ["0", "refused"])
        count, refusal = configured.stdout.split()
        self.assertGreater(int(count), 0)
        self.assertEqual(refusal, "refused")

    def test_configure_raises_once_the_registry_was_read(self):
        registry.block_types()
        with self.assertRaises(gr4.Error):
            gr4.configure(plugin_dirs=os.environ["GNURADIO4_PLUGIN_DIRECTORIES"])


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
