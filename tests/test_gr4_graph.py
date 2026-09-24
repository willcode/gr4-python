"""Tests of graphs built by registry key: blocks, settings at construction, ports, edges and documents.

Every graph that writes a file bounds its samples with a head block and writes into a temporary directory.
"""

import array
import os
import re
import sys
import tempfile
import unittest

import gr4
from gr4 import _gr4

SIGNAL_GENERATOR = "gr::blocks::basic::SignalGenerator<float32>"
COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
COPY = "gr::blocks::testing::Copy<float32>"
FILE_SINK = "gr::blocks::fileio::BasicFileSink<float32>"
FILE_SOURCE = "gr::blocks::fileio::BasicFileSource<float32>"
NULL_SOURCE = "gr::blocks::testing::NullSource<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
# The signal generator runs free on the wall clock at its sample rate; at this rate a thousand samples take 50 us.
RATE = 20_000_000


def read_float32(path):
    """Returns the float32 samples of a raw file."""
    samples = array.array("f")
    with open(path, "rb") as file:
        samples.frombytes(file.read())
    return samples


def numbered(document):
    """Returns document with the number of every unique name replaced by N."""
    return re.sub(r"#[0-9]+", "#N", document)


def tone_chain(path, samples, **tone_settings):
    """Returns a graph of a tone, a head block of samples and a file sink writing to path, and its three blocks."""
    graph = gr4.Graph()
    tone = graph.add(SIGNAL_GENERATOR, "tone", sample_rate=RATE, tone_frequency=440_000, **tone_settings)
    head = graph.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=samples)
    sink = graph.add(FILE_SINK, "sink", file_name=path)
    graph.connect(tone, head, sink)
    return graph, tone, head, sink


def named(graph, name):
    """Returns the block of graph that carries name."""
    return next(block for block in graph if block.name == name)


def without_parameter(document, key):
    """Returns document without the parameter lines of key; its meta-information lines stay."""
    return "\n".join(line for line in document.splitlines() if not line.strip().startswith(f"{key}:"))


def save_chain(case, samples):
    """Writes the document of a tone chain of samples into the temporary directory of case and returns its path."""
    graph, *_ = tone_chain(case.path("saved.f32"), samples)
    path = case.path("chain.grc")
    with open(path, "w", encoding="utf-8") as file:
        file.write(graph.to_yaml())
    return path


def subgraph_chain(path):
    """Returns a counting source, a subgraph holding a copy, a head block of 1000 and a file sink writing to path.

    Returns the graph, the copy and the subgraph.
    """
    graph = gr4.Graph()
    source = graph.add(COUNTING_SOURCE, "source")
    inner = graph.add_subgraph("inner")
    copy = inner.interior.add(COPY, "copy")
    inner.interior.export(copy.inputs["in"], "sub_in")
    inner.interior.export(copy.outputs["out"], "sub_out")
    head = graph.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=1000)
    graph.connect(source, inner, head, graph.add(FILE_SINK, "sink", file_name=path))
    return graph, copy, inner


class TemporaryFiles(unittest.TestCase):
    """A test case with a temporary directory that is removed after each test."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def path(self, name):
        """Returns the path of name in the temporary directory."""
        return os.path.join(self.directory, name)


class ConstructionSettings(TemporaryFiles):
    """add() checks keywords against the description and converts each value to its setting's type."""

    def test_sample_rate_20_000_000_reaches_a_float32_setting(self):
        graph = gr4.Graph()
        tone = graph.add(SIGNAL_GENERATOR, "tone", sample_rate=20_000_000)
        self.assertEqual(tone.settings.types["sample_rate"], "float32")
        self.assertEqual(tone.settings["sample_rate"], 20_000_000.0)
        # Without the description's type code the settings code refuses the same integer.
        with self.assertRaises(gr4.SettingError):
            graph.add(SIGNAL_GENERATOR, "unconverted", strict=False, sample_rate=20_000_000)

    def test_a_misspelled_keyword_is_refused_with_a_suggestion_and_adds_nothing(self):
        graph = gr4.Graph()
        with self.assertRaises(KeyError) as caught:
            graph.add(SIGNAL_GENERATOR, "tone", tone_frequncy=440.0)
        self.assertIn("'tone_frequency'", str(caught.exception))
        self.assertEqual(len(graph), 0)

    def test_a_value_that_does_not_fit_its_setting_names_the_key_and_the_type(self):
        graph = gr4.Graph()
        with self.assertRaisesRegex(OverflowError, r"'n_samples_max'.*uint32"):
            graph.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=-1)
        with self.assertRaisesRegex(TypeError, r"'tone_frequency'.*float32"):
            graph.add(SIGNAL_GENERATOR, "tone", tone_frequency="high")
        self.assertEqual(len(graph), 0)

    def test_an_unknown_type_is_refused_with_or_without_strict(self):
        graph = gr4.Graph()
        for strict in (True, False):
            with self.subTest(strict=strict), self.assertRaises(gr4.UnknownBlockType):
                graph.add("gr::blocks::testing::NoSuchBlock<float32>", "missing", strict=strict)

    def test_meta_is_merged_into_the_meta_information(self):
        graph = gr4.Graph()
        tone = graph.add(gr4.registry[SIGNAL_GENERATOR], "tone", meta={"origin": "test"})
        self.assertEqual(tone.meta["origin"], "test")
        self.assertEqual(tone.meta["tone_frequency::unit"], "Hz")


class Wiring(TemporaryFiles):
    """connect() takes ports or blocks with one stream port, and edges, remove() and subgraphs follow."""

    def test_a_chain_connects_each_block_to_the_next_with_the_buffer_size_given(self):
        graph = gr4.Graph()
        source = graph.add(NULL_SOURCE, "source")
        copy = graph.add(COPY, "copy")
        sink = graph.add(NULL_SINK, "sink")
        graph.connect(source.outputs["out"], copy, sink.inputs["in"], buffer_size=8192, name="chain")
        edges = {(edge.source, edge.source_port, edge.destination, edge.destination_port) for edge in graph.edges}
        self.assertEqual(edges, {(source, "out", copy, "in"), (copy, "out", sink, "in")})
        self.assertEqual({edge.buffer_size for edge in graph.edges}, {8192})
        self.assertEqual({edge.name for edge in graph.edges}, {"chain"})
        graph.disconnect(copy, sink)
        self.assertEqual([(edge.source, edge.destination) for edge in graph.edges], [(source, copy)])

    def test_an_end_that_names_no_single_port_in_its_direction_is_refused(self):
        graph = gr4.Graph()
        source = graph.add(NULL_SOURCE, "source")
        sink = graph.add(NULL_SINK, "sink")
        empty = graph.add_subgraph("empty")
        with self.assertRaisesRegex(gr4.PortError, "0 input stream ports"):
            graph.connect(source, empty)
        with self.assertRaisesRegex(gr4.PortError, "is an input"):
            graph.connect(sink.inputs["in"], source)
        with self.assertRaisesRegex(gr4.PortError, "is an input"):
            graph.connect(source, sink.inputs["in"], sink)
        self.assertEqual(graph.edges, [])
        self.assertIsInstance(gr4.PortError("x"), LookupError)

    def test_a_removed_block_reports_invalid_and_leaves_the_graph(self):
        graph = gr4.Graph()
        source = graph.add(NULL_SOURCE, "source")
        sink = graph.add(NULL_SINK, "sink")
        graph.connect(source, sink)
        self.assertTrue(sink.valid)
        graph.remove(sink)
        self.assertFalse(sink.valid)
        self.assertEqual(list(graph), [source])
        self.assertEqual(graph.edges, [])
        self.assertEqual(sink.name, "sink")

    def test_an_exported_interior_port_appears_on_the_subgraph(self):
        graph, copy, inner = subgraph_chain(self.path("unused.f32"))
        self.assertEqual(list(inner.inputs), ["sub_in"])
        self.assertEqual(list(inner.outputs), ["sub_out"])
        self.assertEqual(len(graph.edges), 3)
        self.assertIn(copy, graph.blocks(recursive=True))
        self.assertNotIn(copy, graph.blocks())
        self.assertEqual(graph.find(copy.unique_name), copy)
        inner.interior.unexport(copy.outputs["out"])
        self.assertEqual(list(inner.outputs), [])

    def test_ports_of_different_value_types_are_refused_naming_both_and_the_chain_adds_no_edge(self):
        graph = gr4.Graph()
        source = graph.add(COUNTING_SOURCE, "source")
        copy = graph.add(COPY, "copy")
        head = graph.add("gr::blocks::testing::HeadBlock<int16>", "head")
        self.assertEqual((copy.outputs["out"].type, head.inputs["in"].type), ("float32", "int16"))
        with self.assertRaisesRegex(gr4.PortError, r"'copy' carries float32, and input 'in' of 'head' takes int16"):
            graph.connect(source, copy, head)
        self.assertEqual(graph.edges, [])

    def test_every_chain_of_one_value_type_connects(self):
        def ports(build):
            edges = build().edges
            return len(edges), {
                (edge.source.outputs[edge.source_port].type, edge.destination.inputs[edge.destination_port].type)
                for edge in edges
            }

        def two(source, destination):
            graph = gr4.Graph()
            graph.connect(graph.add(source, "source"), graph.add(destination, "sink"))
            return graph

        def by_port():
            graph = gr4.Graph()
            source, copy, sink = graph.add(NULL_SOURCE, "source"), graph.add(COPY, "copy"), graph.add(NULL_SINK, "sink")
            graph.connect(source.outputs["out"], copy, sink.inputs["in"], buffer_size=8192)
            return graph

        def counted():
            graph = gr4.Graph()
            head = graph.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=1000)
            graph.connect(graph.add(COUNTING_SOURCE, "source"), head, graph.add(FILE_SINK, "sink"))
            return graph

        chains = {
            "tone, head, file sink": (lambda: tone_chain(self.path("unused.f32"), 1000)[0], 2),
            "counting source, subgraph, head, file sink": (lambda: subgraph_chain(self.path("unused.f32"))[0], 3),
            "null source, copy, null sink by port": (by_port, 2),
            "counting source, head, file sink": (counted, 2),
            "tone, null sink": (lambda: two(SIGNAL_GENERATOR, NULL_SINK), 1),
            "null source, null sink": (lambda: two(NULL_SOURCE, NULL_SINK), 1),
            "file source, null sink": (lambda: two(FILE_SOURCE, NULL_SINK), 1),
        }
        for name, (build, edges) in chains.items():
            with self.subTest(chain=name):
                self.assertEqual(ports(build), (edges, {("float32", "float32")}))


class Documents(TemporaryFiles):
    """A graph document survives a save and a load, and settings merged into it are in force once it loads."""

    def test_a_document_survives_a_load_and_a_save(self):
        graph, *_ = tone_chain(self.path("unused.f32"), 1000)
        document = graph.to_yaml()
        loaded = gr4.Graph.from_yaml(document)
        # The loader numbers the unique names of the blocks it creates anew.
        self.assertEqual(numbered(loaded.to_yaml()), numbered(document))
        self.assertEqual(sorted(block.name for block in loaded), ["head", "sink", "tone"])
        self.assertEqual(len(loaded.edges), 2)

    def test_settings_given_to_load_are_in_force_when_the_graph_loads(self):
        saved = save_chain(self, 1000)
        loaded = gr4.Graph.load(saved, settings={"head": {"n_samples_max": 777}})
        self.assertEqual(named(loaded, "head").settings["n_samples_max"], 777)
        self.assertEqual(named(loaded, "head").settings.types["n_samples_max"], "uint32")
        with open(saved, encoding="utf-8") as file:
            self.assertEqual(named(gr4.Graph.load(file), "head").settings["n_samples_max"], 1000)

    def test_a_load_setting_a_type_does_not_declare_is_refused_with_the_closest_keys(self):
        graph, *_ = tone_chain(self.path("unused.f32"), 1000)
        graph.add(gr4.DEFAULT_SCHEDULER, "nested")
        document = graph.to_yaml()
        with self.assertRaisesRegex(KeyError, r"'tone_frequncy'.*'tone_frequency'"):
            gr4.Graph.from_yaml(document, settings={"tone": {"amplitude": 0.5, "tone_frequncy": 5.0}})
        with self.assertRaisesRegex(KeyError, r"'timeout_m'.*'timeout_ms'"):
            gr4.Graph.from_yaml(document, settings={"nested": {"timeout_m": 5}})
        loaded = gr4.Graph.from_yaml(document, settings={"tone": {"tone_frequency": 5.0}, "nested": {"timeout_ms": 5}})
        self.assertEqual(named(loaded, "tone").settings["tone_frequency"], 5.0)
        self.assertEqual(named(loaded, "nested").settings["timeout_ms"], 5)

    def test_a_load_setting_the_document_does_not_hold_takes_the_type_its_setting_declares(self):
        tone = gr4.Graph()
        tone.add(SIGNAL_GENERATOR, "tone")
        product = gr4.Graph()
        product.add(gr4.registry.find("MultiplyConst", "complex<float32>"), "product")
        cases = {
            "tone": (without_parameter(tone.to_yaml(), "sample_rate"), "sample_rate", 20_000_000, 20_000_000.0),
            "product": (without_parameter(product.to_yaml(), "value"), "value", 2 + 1j, 2 + 1j),
        }
        for name, (document, key, value, loaded) in cases.items():
            with self.subTest(block=name):
                self.assertIsNone(re.search(rf"^\s*{key}:", document, re.MULTILINE))
                # Merged with the type its Python value selects, the same value is refused.
                with self.assertRaises(gr4.Error):
                    _gr4.RuntimeGraph.from_yaml(_gr4.merge_parameters(document, {name: {key: value}}))
                graph = gr4.Graph.from_yaml(document, settings={name: {key: value}})
                self.assertEqual(named(graph, name).settings[key], loaded)

    def test_a_malformed_document_names_the_line_and_column(self):
        with self.assertRaisesRegex(gr4.GraphFileError, r"line \d+, column \d+") as caught:
            gr4.Graph.from_yaml('blocks:\n  - id: "unterminated\n')
        self.assertIsInstance(caught.exception, ValueError)
        with self.assertRaisesRegex(gr4.GraphFileError, "no block of the document is named 'absent'"):
            gr4.Graph.from_yaml(gr4.Graph().to_yaml(), settings={"absent": {"x": 1}})


class Runs(TemporaryFiles):
    """A graph built here runs to the end: a head block of N samples writes N samples."""

    def test_a_head_block_of_n_samples_writes_n_samples_of_the_tone(self):
        for samples in (1000, 4096 + 3):
            with self.subTest(samples=samples):
                path = self.path(f"tone-{samples}.f32")
                graph, *_ = tone_chain(path, samples, amplitude=0.5)
                graph.run()
                written = read_float32(path)
                self.assertEqual(len(written), samples)
                self.assertLessEqual(max(abs(sample) for sample in written), 0.5)
                self.assertGreater(max(abs(sample) for sample in written), 0.25)

    def test_an_exported_interior_port_carries_samples_through_a_subgraph(self):
        direct, through = self.path("direct.f32"), self.path("through.f32")
        graph = gr4.Graph()
        head = graph.add(gr4.registry.find("HeadBlock", "float32"), "head", n_samples_max=1000)
        graph.connect(graph.add(COUNTING_SOURCE, "source"), head, graph.add(FILE_SINK, "sink", file_name=direct))
        graph.run()
        subgraph_chain(through)[0].run()
        self.assertEqual(len(read_float32(through)), 1000)
        self.assertEqual(read_float32(through), read_float32(direct))

    def test_settings_given_to_load_are_in_force_at_start(self):
        merged = self.path("merged.f32")
        loaded = gr4.Graph.load(
            save_chain(self, 1000), settings={"head": {"n_samples_max": 777}, "sink": {"file_name": merged}}
        )
        loaded.run()
        self.assertEqual(len(read_float32(merged)), 777)
        self.assertFalse(os.path.exists(self.path("saved.f32")))


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
