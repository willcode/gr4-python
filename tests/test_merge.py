"""Tests of settings merged into a graph document before the graph loads."""

import difflib
import os
import sys
import tempfile
import unittest

from gr4 import _gr4

COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"
FILE_SINK = "gr::blocks::fileio::BasicFileSink<float32>"
COPY = "gr::blocks::testing::Copy<float32>"
FLOAT32_BYTES = 4
WRITTEN_SAMPLES = 10
MERGED_SAMPLES = 3000


def counted_chain(path):
    """Returns a graph of a counting source, a head block of WRITTEN_SAMPLES and a file sink writing to path."""
    graph = _gr4.RuntimeGraph()
    source = graph.emplace(COUNTING_SOURCE, "source")
    head = graph.emplace(HEAD, "head", {"n_samples_max": WRITTEN_SAMPLES}, {"n_samples_max": "uint32"})
    sink = graph.emplace(FILE_SINK, "sink", {"file_name": path})
    graph.connect(source, "out", head, "in")
    graph.connect(head, "out", sink, "in")
    return graph


def changed_lines(before, after):
    """Returns the lines a unified diff removes and adds, without their markers."""
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
    lines = [line for line in diff if not line.startswith(("---", "+++", "@@"))]
    return [line[1:] for line in lines if line.startswith("-")], [line[1:] for line in lines if line.startswith("+")]


class MergeIntoDocument(unittest.TestCase):
    """A merged setting is in force when its block starts, and the rest of the document is unchanged."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, "merged.f32")
        self.document = counted_chain(self.path).to_yaml()

    def test_a_merged_head_block_length_is_in_force_when_the_block_starts(self):
        merged = _gr4.merge_parameters(self.document, {"head": {"n_samples_max": MERGED_SAMPLES}})
        runtime = _gr4.Runtime.create(_gr4.RuntimeGraph.from_yaml(merged))
        runtime.run_and_wait()
        runtime.close()
        self.assertEqual(os.path.getsize(self.path), MERGED_SAMPLES * FLOAT32_BYTES)

    def test_only_the_merged_value_changes_and_it_keeps_the_written_type(self):
        merged = _gr4.merge_parameters(self.document, {"head": {"n_samples_max": MERGED_SAMPLES}})
        removed, added = changed_lines(self.document, merged)
        self.assertEqual([line.strip() for line in removed], [f"n_samples_max: !!uint32 {WRITTEN_SAMPLES}"])
        self.assertEqual([line.strip() for line in added], [f"n_samples_max: !!uint32 {MERGED_SAMPLES}"])
        self.assertIn("max_bytes_per_file: !!uint32 0", merged)

    def test_a_block_is_found_by_its_unique_name(self):
        document = (
            "blocks:\n"
            "  - id: gr::blocks::testing::HeadBlock<float32>\n"
            "    unique_name: head-17\n"
            "    parameters:\n"
            f"      n_samples_max: !!uint32 {WRITTEN_SAMPLES}\n"
        )
        merged = _gr4.merge_parameters(document, {"head-17": {"n_samples_max": MERGED_SAMPLES}})
        self.assertIn(f"n_samples_max: !!uint32 {MERGED_SAMPLES}", merged)

    def test_a_new_parameter_takes_the_type_code_given_or_the_plain_mapping(self):
        typed = _gr4.merge_parameters(self.document, {"head": {"marker": 5}}, {"head": {"marker": "int16"}})
        self.assertIn("marker: !!int16 5", typed)
        plain = _gr4.merge_parameters(self.document, {"head": {"marker": 5}})
        self.assertIn("marker: !!int64 5", plain)

    def test_a_value_that_does_not_fit_the_written_type_names_its_key(self):
        with self.assertRaisesRegex(OverflowError, "'n_samples_max'"):
            _gr4.merge_parameters(self.document, {"head": {"n_samples_max": -1}})

    def test_a_name_no_block_carries_raises_error_naming_the_blocks_there_are(self):
        with self.assertRaises(_gr4.Error) as caught:
            _gr4.merge_parameters(self.document, {"tail": {"n_samples_max": 1}})
        for name in ("tail", "source", "head", "sink"):
            self.assertIn(f"'{name}'", caught.exception.message)

    def test_a_name_two_blocks_carry_raises_error(self):
        graph = _gr4.RuntimeGraph()
        graph.emplace(HEAD, "twin")
        graph.emplace(HEAD, "twin")
        with self.assertRaisesRegex(_gr4.Error, "2 blocks of the document are named 'twin'"):
            _gr4.merge_parameters(graph.to_yaml(), {"twin": {"n_samples_max": 1}})

    def test_a_block_inside_a_subgraph_is_found(self):
        graph = _gr4.RuntimeGraph()
        subgraph = graph.emplace_subgraph("inner")
        graph.interior(subgraph).emplace(COPY, "inner_copy")
        document = graph.to_yaml()
        self.assertIn("inner_copy", document)
        merged = _gr4.merge_parameters(document, {"inner_copy": {"marker": 7}}, {"inner_copy": {"marker": "int16"}})
        self.assertIn("marker: !!int16 7", merged)

    def test_the_settings_of_a_nested_scheduler_are_in_force_when_the_graph_loads(self):
        graph = _gr4.RuntimeGraph()
        written = graph.emplace(_gr4.DEFAULT_SCHEDULER, "nested").get_value("timeout_ms")
        merged = _gr4.merge_parameters(
            graph.to_yaml(), {"nested": {"timeout_ms": written + 77}}, {"nested": {"timeout_ms": "uint32"}}
        )
        self.assertIn(f"timeout_ms: !!uint32 {written + 77}", merged)
        (loaded,) = [block for block in _gr4.RuntimeGraph.from_yaml(merged).blocks() if block.name == "nested"]
        self.assertEqual(loaded.get_value("timeout_ms"), written + 77)

    def test_a_subgraph_is_found_by_its_top_level_name_and_takes_settings_only_with_a_scheduler(self):
        document = (
            "blocks:\n"
            "  - id: SUBGRAPH\n"
            "    name: nested\n"
            "    scheduler:\n"
            f"      id: {_gr4.DEFAULT_SCHEDULER}\n"
            "    graph:\n"
            "      blocks: []\n"
            "  - id: SUBGRAPH\n"
            "    name: plain\n"
            "    graph:\n"
            "      blocks: []\n"
        )
        merged = _gr4.merge_parameters(document, {"nested": {"timeout_ms": 5}}, {"nested": {"timeout_ms": "uint32"}})
        self.assertIn("timeout_ms: !!uint32 5", merged)
        with self.assertRaisesRegex(_gr4.Error, "'plain' is a subgraph without a scheduler"):
            _gr4.merge_parameters(document, {"plain": {"marker": 1}})

    def test_a_document_that_does_not_parse_names_the_line_and_column(self):
        malformed = 'blocks:\n  - id: gr::blocks::testing::NullSink<float32>\n    parameters:\n      name: "sink\n'
        with self.assertRaisesRegex(_gr4.Error, r"line \d+, column \d+"):
            _gr4.merge_parameters(malformed, {"sink": {}})


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
