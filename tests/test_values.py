"""Tests of the conversion between Python objects and pmt values."""

import array
import ctypes
import math
import os
import struct
import sys
import unittest

from gr4 import _gr4

FLOAT32_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
FLOAT32_BELOW_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFE))[0]
FLOAT32_SMALLEST = struct.unpack("<f", struct.pack("<I", 0x00000001))[0]
# A double at or above FLOAT32_ROUNDING_LIMIT rounds to infinity; the double below it rounds to FLOAT32_MAX.
FLOAT32_ROUNDING_LIMIT = FLOAT32_MAX + (FLOAT32_MAX - FLOAT32_BELOW_MAX) / 2

INTEGER_BOUNDS = {
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}


def as_float32(number):
    """Returns number rounded to the nearest float32, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", number))[0]


class ScalarBounds(unittest.TestCase):
    """Every scalar type code holds its bounds and refuses the first value past them."""

    def test_integer_codes_round_trip_at_their_bounds_and_refuse_one_past(self):
        for code, (lowest, highest) in INTEGER_BOUNDS.items():
            with self.subTest(code=code):
                for bound in (lowest, highest):
                    self.assertEqual(_gr4.convert(bound, code), bound)
                    self.assertEqual(_gr4.type_code(bound, code), code)
                for past in (lowest - 1, highest + 1):
                    with self.assertRaises(OverflowError):
                        _gr4.convert(past, code)

    def test_float32_rounds_to_its_bounds_and_refuses_a_value_that_rounds_past_them(self):
        for bound in (FLOAT32_MAX, -FLOAT32_MAX, FLOAT32_SMALLEST):
            self.assertEqual(_gr4.convert(bound, "float32"), bound)
        below_limit = math.nextafter(FLOAT32_ROUNDING_LIMIT, 0.0)
        self.assertEqual(as_float32(below_limit), FLOAT32_MAX)
        with self.assertRaises(OverflowError):
            as_float32(FLOAT32_ROUNDING_LIMIT)
        for rounded in (math.nextafter(FLOAT32_MAX, math.inf), below_limit):
            self.assertEqual(_gr4.convert(rounded, "float32"), FLOAT32_MAX)
            self.assertEqual(_gr4.convert(-rounded, "float32"), -FLOAT32_MAX)
        for past in (FLOAT32_ROUNDING_LIMIT, -FLOAT32_ROUNDING_LIMIT, 2**128):
            with self.assertRaises(OverflowError):
                _gr4.convert(past, "float32")
        self.assertEqual(_gr4.convert(math.inf, "float32"), math.inf)

    def test_float64_round_trips_at_its_bounds_and_refuses_an_integer_past_them(self):
        for bound in (sys.float_info.max, -sys.float_info.max, math.ulp(0.0)):
            self.assertEqual(_gr4.convert(bound, "float64"), bound)
        with self.assertRaises(OverflowError):
            _gr4.convert(2**1024, "float64")

    def test_complex_codes_round_trip_at_their_bounds_and_refuse_one_past(self):
        bounds = {"complex<float32>": FLOAT32_MAX, "complex<float64>": sys.float_info.max}
        for code, bound in bounds.items():
            with self.subTest(code=code):
                self.assertEqual(_gr4.convert(complex(bound, -bound), code), complex(bound, -bound))
        below_limit = math.nextafter(FLOAT32_ROUNDING_LIMIT, 0.0)
        self.assertEqual(_gr4.convert(complex(0.0, below_limit), "complex<float32>"), complex(0.0, FLOAT32_MAX))
        with self.assertRaises(OverflowError):
            _gr4.convert(complex(0.0, FLOAT32_ROUNDING_LIMIT), "complex<float32>")

    def test_bool_holds_both_values_and_refuses_integers(self):
        for flag in (True, False):
            self.assertIs(_gr4.convert(flag, "bool"), flag)
        with self.assertRaises(TypeError):
            _gr4.convert(1, "bool")
        with self.assertRaises(TypeError):
            _gr4.convert(True, "int8")

    def test_string_holds_any_text_and_refuses_bytes(self):
        for text in ("", "gain", "Grüße ✓"):
            self.assertEqual(_gr4.convert(text, "string"), text)
        with self.assertRaises(TypeError):
            _gr4.convert(b"gain", "string")

    def test_a_lone_surrogate_outside_the_escape_range_is_a_type_error_naming_its_key(self):
        escaped = "raw\udc80byte"
        self.assertEqual(_gr4.convert(escaped, "string"), escaped)
        with self.assertRaisesRegex(TypeError, "'label'") as caught:
            _gr4.convert({"label": "bad\ud800"})
        self.assertIsInstance(caught.exception.__cause__, UnicodeEncodeError)
        with self.assertRaises(TypeError):
            _gr4.convert("\ud800", "string")

    def test_none_holds_only_none(self):
        self.assertIsNone(_gr4.convert(None, "none"))
        self.assertIsNone(_gr4.convert(None))
        self.assertEqual(_gr4.type_code(None), "none")
        with self.assertRaises(TypeError):
            _gr4.convert(0, "none")


class TargetTypes(unittest.TestCase):
    """A type code coerces where the settings code of the core would refuse."""

    def test_python_complex_converts_to_complex_float32(self):
        self.assertEqual(_gr4.convert(1.5 - 2.25j, "complex<float32>"), 1.5 - 2.25j)
        self.assertEqual(_gr4.type_code(1.5 - 2.25j, "complex<float32>"), "complex<float32>")
        self.assertEqual(_gr4.type_code(1.5 - 2.25j), "complex<float64>")

    def test_an_integer_converts_to_float32_as_the_same_number_written_as_a_float(self):
        self.assertEqual(_gr4.type_code(20_000_000, "float32"), "float32")
        self.assertEqual(_gr4.convert(20_000_000, "float32"), 20_000_000.0)
        self.assertEqual(_gr4.convert(2**24 + 1, "float32"), as_float32(float(2**24 + 1)))
        self.assertEqual(_gr4.convert(0.1, "float32"), as_float32(0.1))

    def test_a_fraction_is_refused_for_an_integer_type(self):
        with self.assertRaisesRegex(OverflowError, "fractional"):
            _gr4.convert(2.5, "int32")
        whole = _gr4.convert(3.0, "int32")
        self.assertEqual(whole, 3)
        self.assertIs(type(whole), int)

    def test_an_unknown_type_code_is_a_value_error(self):
        for code in ("float128", "tensor<>", "value", "tensor<map>"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                _gr4.convert(1, code)


class Tensors(unittest.TestCase):
    """Lists, tuples and buffers become tensors, and tensors become lists."""

    def test_a_tensor_of_each_numeric_type_round_trips_at_its_bounds(self):
        bounds = dict(INTEGER_BOUNDS)
        bounds["float32"] = (-FLOAT32_MAX, FLOAT32_MAX)
        bounds["float64"] = (-sys.float_info.max, sys.float_info.max)
        bounds["complex<float32>"] = (complex(-FLOAT32_MAX, 1.0), complex(0.5, FLOAT32_MAX))
        bounds["complex<float64>"] = (complex(-sys.float_info.max, 1.0), complex(0.5, sys.float_info.max))
        bounds["bool"] = (False, True)
        for code, (lowest, highest) in bounds.items():
            with self.subTest(code=code):
                tensor_code = f"tensor<{code}>"
                self.assertEqual(_gr4.convert([lowest, highest], tensor_code), [lowest, highest])
                self.assertEqual(_gr4.convert((lowest, highest), tensor_code), [lowest, highest])
                self.assertEqual(_gr4.type_code([lowest, highest], tensor_code), tensor_code)

    def test_an_element_out_of_range_is_refused_and_named(self):
        with self.assertRaisesRegex(OverflowError, "element 1"):
            _gr4.convert([0, 256], "tensor<uint8>")
        with self.assertRaisesRegex(TypeError, "element 2"):
            _gr4.convert([1.0, 2.0, "3"], "tensor<float32>")

    def test_nested_lists_are_a_tensor_of_higher_rank(self):
        rows = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(_gr4.convert(rows, "tensor<int16>"), rows)
        with self.assertRaisesRegex(TypeError, "ragged"):
            _gr4.convert([[1, 2], [3]], "tensor<int16>")

    def test_strings_are_a_tensor_of_strings(self):
        self.assertEqual(_gr4.type_code(["a", "b"]), "tensor<string>")
        self.assertEqual(_gr4.convert(["a", "b"]), ["a", "b"])
        self.assertEqual(_gr4.convert(("a", "b"), "tensor<string>"), ["a", "b"])
        with self.assertRaises(TypeError):
            _gr4.convert(["a", 1], "tensor<string>")

    def test_mixed_elements_are_a_tensor_of_type_erased_values(self):
        mixed = [1, "a", 2.5, None, [2, 3], {"k": True}]
        self.assertEqual(_gr4.type_code(mixed), "tensor<value>")
        self.assertEqual(_gr4.convert(mixed), mixed)
        self.assertEqual(_gr4.type_code([]), "tensor<value>")
        self.assertEqual(_gr4.convert([], "tensor<float32>"), [])

    def test_a_buffer_is_the_typed_tensor_of_its_format(self):
        for typecode in "bBhHiIlLqQ":
            with self.subTest(typecode=typecode):
                bits = 8 * array.array(typecode).itemsize
                code = f"{'int' if typecode.islower() else 'uint'}{bits}"
                lowest, highest = INTEGER_BOUNDS[code]
                numbers = array.array(typecode, [lowest, 0, highest])
                self.assertEqual(_gr4.type_code(numbers), f"tensor<{code}>")
                self.assertEqual(_gr4.convert(numbers), [lowest, 0, highest])
        self.assertEqual(_gr4.type_code(array.array("f", [1.5])), "tensor<float32>")
        self.assertEqual(_gr4.convert(array.array("f", [1.5, -FLOAT32_MAX])), [1.5, -FLOAT32_MAX])
        self.assertEqual(_gr4.type_code(array.array("d", [1.5])), "tensor<float64>")
        self.assertEqual(_gr4.convert(memoryview(bytes([1, 0, 1])).cast("?")), [True, False, True])
        self.assertEqual(_gr4.type_code(b"\x00\x01"), "tensor<uint8>")
        self.assertEqual(_gr4.convert(memoryview(bytes(range(6))).cast("B", (2, 3))), [[0, 1, 2], [3, 4, 5]])

    def test_a_complex_buffer_is_a_complex_tensor(self):
        if not hasattr(ctypes, "c_float_complex"):
            self.skipTest("ctypes has complex types from Python 3.14")
        values = [1 + 2j, -3.5j]
        self.assertEqual(_gr4.type_code((ctypes.c_float_complex * 2)(*values)), "tensor<complex<float32>>")
        self.assertEqual(_gr4.convert((ctypes.c_double_complex * 2)(*values)), values)

    def test_a_buffer_converts_element_by_element_to_a_named_type(self):
        self.assertEqual(_gr4.convert(array.array("d", [1.0, 2.0]), "tensor<int32>"), [1, 2])
        with self.assertRaisesRegex(OverflowError, "element 0"):
            _gr4.convert(array.array("d", [1.5]), "tensor<int32>")

    def test_a_zero_dimensional_buffer_is_the_scalar_of_its_format(self):
        scalar = memoryview(struct.pack("f", 1.5)).cast("f", shape=[])
        self.assertEqual(_gr4.type_code(scalar), "float32")
        self.assertEqual(_gr4.convert(scalar, "float64"), 1.5)
        flag = memoryview(b"\x01").cast("?", shape=[])
        self.assertIs(_gr4.convert(flag, "bool"), True)

    def test_a_buffer_in_foreign_byte_order_is_refused(self):
        with self.assertRaises(TypeError):
            _gr4.convert((ctypes.c_int32.__ctype_be__ * 2)())


class Maps(unittest.TestCase):
    """A dict with str keys is a map, at any depth."""

    def test_maps_nested_in_maps_round_trip(self):
        nested = {"a": {"b": {"c": 1, "d": [1.5, "x"]}}, "e": None, "f": "text"}
        self.assertEqual(_gr4.type_code(nested), "map")
        self.assertEqual(_gr4.convert(nested), nested)
        self.assertEqual(_gr4.convert({}, "map"), {})

    def test_a_key_that_is_not_a_string_is_refused(self):
        with self.assertRaises(TypeError):
            _gr4.convert({1: 2})


class Nesting(unittest.TestCase):
    """An object nested past the interpreter's recursion limit raises RecursionError."""

    def test_a_deep_or_self_referencing_object_raises_recursion_error(self):
        deep = []
        for _ in range(100_000):
            deep = [deep]
        looped = []
        looped.append(looped)
        cyclic = {}
        cyclic["self"] = cyclic
        cases = {
            "deep list": (deep, None),
            "deep list to a tensor": (deep, "tensor<int8>"),
            "list holding itself": (looped, None),
            "list holding itself to a tensor": (looped, "tensor<int8>"),
            "dict holding itself": (cyclic, None),
            "dict holding itself to a map": (cyclic, "map"),
        }
        for name, (obj, code) in cases.items():
            with self.subTest(case=name), self.assertRaises(RecursionError):
                _gr4.convert(obj, code)
        self.assertEqual(_gr4.convert([[[1]]]), [[[1]]])


class PythonTypes(unittest.TestCase):
    """Without a type code the Python type selects the value type."""

    def test_each_python_type_selects_its_value_type(self):
        cases = [
            (True, "bool"),
            (1, "int64"),
            (-(2**63), "int64"),
            (2**63 - 1, "int64"),
            (2**63, "uint64"),
            (2**64 - 1, "uint64"),
            (1.5, "float64"),
            (1j, "complex<float64>"),
            ("text", "string"),
            ({}, "map"),
        ]
        for obj, code in cases:
            with self.subTest(obj=obj):
                self.assertEqual(_gr4.type_code(obj), code)
                self.assertEqual(_gr4.convert(obj), obj)

    def test_an_integer_beyond_uint64_or_below_int64_is_refused(self):
        for number in (2**64, -(2**63) - 1):
            with self.subTest(number=number), self.assertRaises(OverflowError):
                _gr4.convert(number)

    def test_an_object_without_a_value_type_is_refused(self):
        with self.assertRaises(TypeError):
            _gr4.convert(object())


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
