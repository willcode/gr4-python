"""Tests of the settings mapping and of the coercion of Python values to a setting's type."""

import os
import sys
import unittest

import gr4
from gr4 import _gr4
from gr4.values import Settings, coercion_types

SIGNAL_GENERATOR = "gr::blocks::basic::SignalGenerator<float32>"
HEAD = "gr::blocks::testing::HeadBlock<float32>"


class Coercion(unittest.TestCase):
    """Staging converts each value to its setting's type; the settings code alone refuses some of those values."""

    def setUp(self):
        self.graph = _gr4.RuntimeGraph()
        self.tone = self.graph.emplace(SIGNAL_GENERATOR, "tone")
        self.settings = Settings(self.tone, "tone")

    def test_an_integer_wider_than_a_float32_mantissa_is_staged_on_a_float32_setting(self):
        self.assertEqual(self.settings.types["sample_rate"], "float32")
        with self.assertRaises(gr4.Error):
            self.tone.set_staged({"sample_rate": 20_000_001})
        self.settings["sample_rate"] = 20_000_001
        self.assertEqual(self.settings.staged["sample_rate"], 20_000_000.0)

    def test_a_value_of_the_wrong_kind_or_out_of_range_names_the_key_and_the_type(self):
        with self.assertRaisesRegex(TypeError, r"'tone_frequency'.*float32"):
            self.settings["tone_frequency"] = "high"
        head = Settings(self.graph.emplace(HEAD, "head"), "head")
        with self.assertRaisesRegex(OverflowError, r"'n_samples_max'.*uint64"):
            head["n_samples_max"] = -1
        self.assertEqual(self.settings.staged, {})

    def test_an_undeclared_key_is_refused_with_the_closest_keys_and_nothing_is_staged(self):
        with self.assertRaisesRegex(KeyError, "tone_frequency") as caught:
            self.settings.update(amplitude=0.5, tone_frequncy=880.0)
        self.assertIn("tone_frequncy", str(caught.exception))
        self.assertEqual(self.settings.staged, {})
        with self.assertRaisesRegex(KeyError, "sample_rate"):
            self.settings["sample_rat"]

    def test_the_codes_that_admit_no_value_or_only_a_list_are_left_out(self):
        declared = {"a": "float32", "b": "none", "c": "tensor<value>", "d": "tensor<string>"}
        self.assertEqual(coercion_types(declared, ["a", "b", "c", "d", "e"]), {"a": "float32", "d": "tensor<string>"})


class Mapping(unittest.TestCase):
    """The mapping reads the values in force and lists the staged and default values apart."""

    def test_a_staged_value_stays_apart_from_the_value_in_force_until_the_block_applies_it(self):
        graph = _gr4.RuntimeGraph()
        tone = graph.emplace(SIGNAL_GENERATOR, "tone", {"tone_frequency": 440.0}, {"tone_frequency": "float32"})
        settings = Settings(tone, "tone")
        settings.update({"tone_frequency": 880}, amplitude=2)
        self.assertEqual(settings["tone_frequency"], 440.0)
        self.assertEqual(settings.staged, {"tone_frequency": 880.0, "amplitude": 2.0})
        self.assertEqual(settings.defaults["amplitude"], 1.0)
        self.assertEqual(dict(settings), tone.get())
        self.assertEqual(set(settings), set(tone.setting_types()))
        self.assertIn("amplitude", settings)
        self.assertNotIn("amplitud", settings)


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
