"""Tests of graphs built by registry name: blocks, settings, ports and edges."""

import os
import sys
import unittest

from gr4 import _gr4

COUNTING_SOURCE = "gr::blocks::testing::CountingSource<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"
NULL_SINK = "gr::blocks::testing::NullSink<float32>"
COPY = "gr::blocks::testing::Copy<float32>"
ADD = "gr::blocks::math::Add<float32>"
POWER_METER = "gr::blocks::analog::PowerMeter<float32>"
SYMBOL_SYNC = "gr::blocks::sync::SymbolSync<float32>"
FFT = "gr::blocks::fourier::FFT<float32>"


def edge_ends(graph):
    """Returns the source and destination block of every edge of graph."""
    return [(edge.source_block, edge.destination_block) for edge in graph.edges()]


class Chain(unittest.TestCase):
    """A counting source, a head block and a null sink, unconnected until a test connects them."""

    def setUp(self):
        self.graph = _gr4.RuntimeGraph()
        self.source = self.graph.emplace(COUNTING_SOURCE, "source")
        self.head = self.graph.emplace(HEAD, "head", {"n_samples_max": 100}, {"n_samples_max": "uint64"})
        self.sink = self.graph.emplace(NULL_SINK, "sink")

    def test_port_descriptions_agree_with_the_name_lists(self):
        self.assertEqual(self.graph.input_port_names(self.head), ["in"])
        self.assertEqual(self.graph.output_port_names(self.head), ["out"])
        for block in (self.source, self.head, self.sink):
            for is_input, ports, names in (
                (True, self.graph.input_ports(block), self.graph.input_port_names(block)),
                (False, self.graph.output_ports(block), self.graph.output_port_names(block)),
            ):
                with self.subTest(block=block.name, is_input=is_input):
                    self.assertEqual([port.name for port in ports], names)
                    for port in ports:
                        self.assertIs(port.is_input, is_input)
                        self.assertFalse(port.is_message)
                        self.assertEqual(port.type_name, "float32")
                        self.assertFalse(port.is_connected)

    def test_edges_list_what_was_connected_with_the_size_given(self):
        self.graph.connect(self.source, "out", self.head, "in", min_buffer_size=8192, weight=3, name="counted")
        self.graph.connect(self.head, "out", self.sink, "in")
        edges = {edge.name: edge for edge in self.graph.edges()}
        self.assertEqual(len(self.graph.edges()), 2)
        counted = edges["counted"]
        self.assertEqual(counted.source_block, self.source)
        self.assertEqual(counted.source_port, "out")
        self.assertEqual(counted.destination_block, self.head)
        self.assertEqual(counted.destination_port, "in")
        self.assertEqual(counted.min_buffer_size, 8192)
        self.assertEqual(counted.weight, 3)
        (unnamed,) = [edge for edge in self.graph.edges() if edge.name != "counted"]
        self.assertEqual((unnamed.source_block, unnamed.destination_block), (self.head, self.sink))
        self.assertGreater(unnamed.min_buffer_size, 0)
        self.assertFalse(any(port.is_connected for port in self.graph.input_ports(self.head)))

    def test_disconnect_removes_a_recorded_edge_before_a_run_and_after_one(self):
        self.graph.connect(self.source, "out", self.head, "in")
        self.graph.connect(self.head, "out", self.sink, "in")
        self.graph.disconnect(self.head, "out", self.sink, "in")
        self.assertEqual(edge_ends(self.graph), [(self.source, self.head)])
        with self.assertRaises(_gr4.Error) as caught:
            self.graph.disconnect(self.head, "out", self.sink, "in")
        self.assertIn(self.head.unique_name, caught.exception.message)

        self.graph.connect(self.head, "out", self.sink, "in")
        runtime = _gr4.Runtime.create(self.graph)
        self.addCleanup(runtime.close)
        runtime.run_and_wait()
        view = runtime.graph()
        view.disconnect(self.head, "out", self.sink, "in")
        self.assertEqual(edge_ends(view), [(self.source, self.head)])

    def test_a_handle_equals_the_block_found_by_its_unique_name(self):
        found = self.graph.find(self.head.unique_name)
        self.assertEqual(found, self.head)
        self.assertEqual(hash(found), hash(self.head))
        self.assertNotEqual(found, self.source)
        self.assertEqual(set(self.graph.blocks()), {self.source, self.head, self.sink})
        self.assertTrue(self.head.valid)
        self.assertEqual(self.head.name, "head")
        self.assertEqual(self.head.type_name, HEAD)
        for part in ("head", self.head.unique_name, HEAD):
            self.assertIn(part, repr(self.head))

    def test_a_block_reports_its_settings_and_their_types(self):
        self.assertEqual(self.head.get_value("n_samples_max"), 100)
        self.assertEqual(self.head.get(["n_samples_max"]), {"n_samples_max": 100})
        self.assertEqual(self.head.get()["n_samples_max"], 100)
        self.assertEqual(self.head.setting_types()["n_samples_max"], "uint64")
        with self.assertRaises(KeyError):
            self.head.get_value("no_such_setting")

    def test_staging_a_declared_setting_succeeds_and_an_undeclared_one_is_returned(self):
        self.assertEqual(self.head.set_staged({"n_samples_max": 7}, {"n_samples_max": "uint64"}), {})
        self.assertEqual(self.head.staged_parameters()["n_samples_max"], 7)
        self.assertIn("no_such_setting", self.head.set_staged({"no_such_setting": 1.0}))

    def test_a_value_that_does_not_fit_names_its_key(self):
        with self.assertRaisesRegex(OverflowError, "'n_samples_max'"):
            self.head.set_staged({"n_samples_max": -1}, {"n_samples_max": "uint64"})
        with self.assertRaisesRegex(TypeError, "'n_samples_max'"):
            self.head.set_staged({"n_samples_max": "many"}, {"n_samples_max": "uint64"})

    def test_meta_information_is_replaced_and_read_back(self):
        self.head.set_meta_information({"note": "bench", "level": 2})
        information = self.head.meta_information()
        self.assertEqual(information["note"], "bench")
        self.assertEqual(information["level"], 2)

    def test_removing_a_block_leaves_the_others(self):
        self.graph.remove(self.sink)
        self.assertEqual(set(self.graph.blocks()), {self.source, self.head})
        with self.assertRaises(_gr4.Error):
            self.graph.find(self.sink.unique_name)
        for block in self.graph.blocks():
            self.graph.remove(block)
        self.assertEqual(self.graph.blocks(), [])

    def test_a_subgraph_is_wired_through_the_view_of_its_interior(self):
        subgraph = self.graph.emplace_subgraph("inner")
        interior = self.graph.interior(subgraph)
        copy = interior.emplace(COPY, "copy")
        self.assertIn(copy, self.graph.blocks(recursive=True))
        self.assertNotIn(copy, self.graph.blocks())
        interior.export_port(copy, True, "in", "inner_in")
        self.assertIn(copy.unique_name, interior.exported_input_ports())
        self.assertIn("inner_in", self.graph.input_port_names(subgraph))

    def test_an_exported_port_is_listed_until_it_is_unexported(self):
        subgraph = self.graph.emplace_subgraph("inner")
        interior = self.graph.interior(subgraph)
        copy = interior.emplace(COPY, "copy")
        interior.export_port(copy, True, "in", "inner_in")
        interior.export_port(copy, False, "out", "inner_out")
        self.assertEqual(interior.exported_output_ports(), {copy.unique_name: {"out": {"exportedName": "inner_out"}}})
        self.assertEqual(self.graph.output_port_names(subgraph), ["inner_out"])
        interior.unexport_port(copy, True, "in")
        self.assertEqual(interior.exported_input_ports(), {})
        self.assertEqual(self.graph.input_port_names(subgraph), [])
        self.assertEqual(self.graph.output_port_names(subgraph), ["inner_out"])

    def test_the_document_names_every_block_type(self):
        document = self.graph.to_yaml()
        for key in (COUNTING_SOURCE, HEAD, NULL_SINK):
            self.assertIn(key, document)


class PortFields(unittest.TestCase):
    """A port description carries what the port declares."""

    def setUp(self):
        self.graph = _gr4.RuntimeGraph()

    def ports(self, block, is_input):
        """Returns the input or output ports of block by name."""
        listed = self.graph.input_ports(block) if is_input else self.graph.output_ports(block)
        return {port.name: port for port in listed}

    def test_each_element_of_a_port_collection_names_the_collection_and_its_index(self):
        add = self.graph.emplace(ADD, "add", {"n_inputs": 3}, {"n_inputs": "uint32"})
        described = [(port.name, port.collection, port.index) for port in self.graph.input_ports(add)]
        self.assertEqual(described, [(f"in#{index}", "in", index) for index in range(3)])
        (output,) = self.graph.output_ports(add)
        self.assertEqual((output.collection, output.index), ("", 0))

    def test_an_optional_asynchronous_port_says_so_beside_a_required_synchronous_one(self):
        meter = self.graph.emplace(POWER_METER, "meter")
        records = self.ports(meter, False)["records"]
        samples = self.ports(meter, True)["in"]
        self.assertEqual((records.is_optional, records.is_synchronous), (True, False))
        self.assertEqual((samples.is_optional, samples.is_synchronous), (False, True))

    def test_the_sample_bounds_and_the_domain_are_the_ones_each_port_declares(self):
        plain = self.ports(self.graph.emplace(COPY, "copy"), True)["in"]
        symbols = self.ports(self.graph.emplace(SYMBOL_SYNC, "symbols"), True)["in"]
        spectrum = self.ports(self.graph.emplace(FFT, "spectrum"), False)["out"]
        self.assertLess(plain.min_samples, plain.max_samples)
        self.assertGreater(symbols.min_samples, plain.min_samples)
        self.assertEqual(symbols.max_samples, plain.max_samples)
        self.assertLess(spectrum.max_samples, plain.max_samples)
        self.assertEqual(spectrum.min_samples, plain.min_samples)
        for port in (plain, symbols, spectrum):
            self.assertEqual(port.domain, "CPU")


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
