/**
 * @brief The extension module gr4._gr4: the type-erased runtime of gnuradio4-core for Python.
 *
 * The module binds each class of `gnuradio-4.0/Runtime.hpp` as one Python class and merges settings into a graph
 * document. Blocks and schedulers are named by their registry keys, and settings travel as `pmt::Value` maps
 * converted to and from Python objects.
 */

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Runtime.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/YamlPmt.hpp>

#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <expected>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using gr::pmt::Value;

constexpr const char* kPluginDirectoriesVariable = "GNURADIO4_PLUGIN_DIRECTORIES";

/// Names the plugin directory of the core this module was built against when the environment names none. The
/// plugin loader reads the variable once, at the first registry call, and reads the core's installed plugin directory
/// without it. Setting it through `os.environ` gives Python, `plugin_directories()` and the loader one value.
void applyDefaultPluginDirectory() {
    if (std::getenv(kPluginDirectoriesVariable) == nullptr) {
        py::module_::import("os").attr("environ")[kPluginDirectoriesVariable] = GR4_DEFAULT_PLUGIN_DIRECTORY;
    }
}

/// The element type of a scalar or of a tensor, in the order of `Value::ValueType` from `Bool` on. `Value` is a
/// type-erased tensor element.
enum class Element : std::uint8_t { Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, ComplexFloat32, ComplexFloat64, String, Value };

static_assert(static_cast<std::uint8_t>(Element::Bool) + 1U == static_cast<std::uint8_t>(Value::ValueType::Bool));
static_assert(static_cast<std::uint8_t>(Element::Value) + 1U == static_cast<std::uint8_t>(Value::ValueType::Value));

constexpr std::array<std::string_view, 15> kElementNames{"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "complex<float32>", "complex<float64>", "string", "value"};

constexpr std::string_view nameOf(Element element) { return kElementNames[static_cast<std::size_t>(element)]; }

constexpr Element elementOf(Value::ValueType type) { return static_cast<Element>(static_cast<std::uint8_t>(type) - 1U); }

/// A parsed type code: `none`, `map`, a scalar element, or `tensor<element>`, where the element may also be `value`.
struct TypeCode {
    enum class Kind : std::uint8_t { None, Map, Scalar, Tensor };

    Kind    kind    = Kind::None;
    Element element = Element::Value;
};

TypeCode parseTypeCode(std::string_view code) {
    const auto elementNamed = [](std::string_view name) -> std::optional<Element> {
        const auto found = std::ranges::find(kElementNames, name);
        return found == kElementNames.end() ? std::nullopt : std::optional(static_cast<Element>(found - kElementNames.begin()));
    };
    constexpr std::string_view tensorPrefix = "tensor<";
    if (code == "none") {
        return {.kind = TypeCode::Kind::None};
    }
    if (code == "map") {
        return {.kind = TypeCode::Kind::Map};
    }
    if (code.starts_with(tensorPrefix) && code.ends_with('>')) {
        if (const std::optional<Element> element = elementNamed(code.substr(tensorPrefix.size(), code.size() - tensorPrefix.size() - 1UZ)); element.has_value()) {
            return {.kind = TypeCode::Kind::Tensor, .element = *element};
        }
    } else if (const std::optional<Element> element = elementNamed(code); element.has_value() && *element != Element::Value) {
        return {.kind = TypeCode::Kind::Scalar, .element = *element};
    }
    throw py::value_error(std::format("unknown type code '{}'", code));
}

/// The type code of a value. A type-erased tensor whose elements are all strings is `tensor<string>`, the form a
/// string sequence takes in a settings map; any other type-erased tensor is `tensor<value>`.
std::string typeCodeOf(const Value& value) {
    if (value.is_map()) {
        return "map";
    }
    if (value.is_monostate()) {
        return "none";
    }
    if (value.is_tensor()) {
        if (value.value_type() == Value::ValueType::Value) {
            const auto* elements   = value.get_if<gr::Tensor<Value>>();
            const bool  allStrings = elements != nullptr && elements->size() > 0UZ && std::ranges::all_of(*elements, [](const Value& element) { return element.is_string(); });
            return allStrings ? "tensor<string>" : "tensor<value>";
        }
        return std::format("tensor<{}>", nameOf(elementOf(value.value_type())));
    }
    return std::string(nameOf(elementOf(value.value_type())));
}

/// Calls `visit(std::type_identity<T>{})` with the C++ type a Python object converts to for `element`.
template<typename Visitor>
decltype(auto) visitElementType(Element element, Visitor&& visit) {
    switch (element) {
    case Element::Bool: return visit(std::type_identity<bool>{});
    case Element::Int8: return visit(std::type_identity<std::int8_t>{});
    case Element::Int16: return visit(std::type_identity<std::int16_t>{});
    case Element::Int32: return visit(std::type_identity<std::int32_t>{});
    case Element::Int64: return visit(std::type_identity<std::int64_t>{});
    case Element::UInt8: return visit(std::type_identity<std::uint8_t>{});
    case Element::UInt16: return visit(std::type_identity<std::uint16_t>{});
    case Element::UInt32: return visit(std::type_identity<std::uint32_t>{});
    case Element::UInt64: return visit(std::type_identity<std::uint64_t>{});
    case Element::Float32: return visit(std::type_identity<float>{});
    case Element::Float64: return visit(std::type_identity<double>{});
    case Element::ComplexFloat32: return visit(std::type_identity<std::complex<float>>{});
    case Element::ComplexFloat64: return visit(std::type_identity<std::complex<double>>{});
    case Element::String: return visit(std::type_identity<std::pmr::string>{});
    case Element::Value: return visit(std::type_identity<Value>{});
    }
    std::unreachable();
}

/// Identifies the value a conversion error is about: the key of a settings map when there is one, and the element of
/// a tensor.
struct Subject {
    std::string_view           key   = {};
    std::optional<std::size_t> index = std::nullopt;

    [[nodiscard]] Subject element(std::size_t i) const { return {.key = key, .index = i}; }

    [[nodiscard]] std::string describe() const {
        std::string text = key.empty() ? std::string("value") : std::format("'{}'", key);
        if (index.has_value()) {
            text += std::format(" element {}", *index);
        }
        return text;
    }
};

std::string_view typeNameOf(py::handle object) { return Py_TYPE(object.ptr())->tp_name; }

[[noreturn]] void raiseTypeError(const Subject& subject, std::string_view needed, py::handle object) { throw py::type_error(std::format("{} needs {}, got {}", subject.describe(), needed, typeNameOf(object))); }

[[noreturn]] void raiseOverflow(const std::string& message) {
    PyErr_SetString(PyExc_OverflowError, message.c_str());
    throw py::error_already_set();
}

[[noreturn]] void raiseOutOfRange(const Subject& subject, py::handle object, std::string_view needed) { raiseOverflow(std::format("{} {} is out of range for {}", subject.describe(), py::repr(object).cast<std::string>(), needed)); }

/// Throws the Python error a C API call left pending.
void throwIfPythonError() {
    if (PyErr_Occurred() != nullptr) {
        throw py::error_already_set();
    }
}

bool isSequence(py::handle object) { return PyList_Check(object.ptr()) || PyTuple_Check(object.ptr()); }

std::size_t sizeOfSequence(py::handle sequence) { return static_cast<std::size_t>(PyList_Check(sequence.ptr()) ? PyList_GET_SIZE(sequence.ptr()) : PyTuple_GET_SIZE(sequence.ptr())); }

py::object itemOfSequence(py::handle sequence, std::size_t index) {
    const auto i = static_cast<Py_ssize_t>(index);
    return py::reinterpret_borrow<py::object>(PyList_Check(sequence.ptr()) ? PyList_GET_ITEM(sequence.ptr(), i) : PyTuple_GET_ITEM(sequence.ptr(), i));
}

/// An integer, `bool` excluded: a Python `int` or an object with `__index__`.
bool isInteger(py::handle object) { return !PyBool_Check(object.ptr()) && (PyLong_Check(object.ptr()) || PyIndex_Check(object.ptr())); }

/// A Python `float` or an object with `__float__`, `bool` and `complex` excluded; callers test `isInteger` first.
bool isReal(py::handle object) {
    if (PyBool_Check(object.ptr()) || PyComplex_Check(object.ptr())) {
        return false;
    }
    const PyNumberMethods* number = Py_TYPE(object.ptr())->tp_as_number;
    return PyFloat_Check(object.ptr()) || (number != nullptr && number->nb_float != nullptr);
}

py::int_ indexOf(py::handle object) {
    auto index = py::reinterpret_steal<py::int_>(PyNumber_Index(object.ptr()));
    if (!index) {
        throw py::error_already_set();
    }
    return index;
}

std::string stringFrom(py::handle object, const Subject& subject) {
    if (!PyUnicode_Check(object.ptr())) {
        raiseTypeError(subject, "string", object);
    }
    auto encoded = py::reinterpret_steal<py::bytes>(PyUnicode_AsEncodedString(object.ptr(), "utf-8", "surrogateescape"));
    if (!encoded) {
        if (PyErr_ExceptionMatches(PyExc_UnicodeEncodeError) != 0) {
            py::raise_from(PyExc_TypeError, std::format("{} needs a str UTF-8 can encode, got a lone surrogate outside U+DC80..U+DCFF", subject.describe()).c_str());
        }
        throw py::error_already_set();
    }
    return std::string(encoded);
}

/// Holds one level of the interpreter's recursion limit while a conversion descends into a nested object. A deeply
/// nested or self-referencing object raises RecursionError.
class RecursionGuard {
public:
    RecursionGuard() {
        if (Py_EnterRecursiveCall(" while converting a nested object") != 0) {
            throw py::error_already_set();
        }
    }
    ~RecursionGuard() { Py_LeaveRecursiveCall(); }
    RecursionGuard(const RecursionGuard&)            = delete;
    RecursionGuard& operator=(const RecursionGuard&) = delete;
};

py::object stringToPython(std::string_view text) {
    auto decoded = py::reinterpret_steal<py::object>(PyUnicode_DecodeUTF8(text.data(), static_cast<Py_ssize_t>(text.size()), "surrogateescape"));
    if (!decoded) {
        throw py::error_already_set();
    }
    return decoded;
}

/// Integer `T` from a Python integer in range, or from a float holding a whole number in range.
template<std::integral T>
T integerFrom(py::handle object, const Subject& subject, std::string_view code) {
    if (isInteger(object)) {
        const py::int_  number   = indexOf(object);
        int             overflow = 0;
        const long long value    = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
        throwIfPythonError();
        if (overflow == 0) {
            if (!std::in_range<T>(value)) {
                raiseOutOfRange(subject, object, code);
            }
            return static_cast<T>(value);
        }
        if constexpr (std::same_as<T, std::uint64_t>) {
            if (overflow > 0) {
                const unsigned long long unsignedValue = PyLong_AsUnsignedLongLong(number.ptr());
                if (PyErr_Occurred() == nullptr) {
                    return static_cast<T>(unsignedValue);
                }
                PyErr_Clear();
            }
        }
        raiseOutOfRange(subject, object, code);
    }
    if (isReal(object)) {
        const double value = PyFloat_AsDouble(object.ptr());
        throwIfPythonError();
        const double lowest = static_cast<double>(std::numeric_limits<T>::min());
        const double limit  = std::ldexp(1.0, std::numeric_limits<T>::digits);
        if (!std::isfinite(value) || value < lowest || value >= limit) {
            raiseOutOfRange(subject, object, code);
        }
        if (std::trunc(value) != value) {
            raiseOverflow(std::format("{} {} has a fractional part; {} holds whole numbers", subject.describe(), py::repr(object).cast<std::string>(), code));
        }
        return static_cast<T>(value);
    }
    raiseTypeError(subject, code, object);
}

/// A real number as a double. An integer converts the way the same number written as a float literal does.
double realFrom(py::handle object, const Subject& subject, std::string_view code) {
    if (isInteger(object)) {
        const double value = PyLong_AsDouble(indexOf(object).ptr());
        if (value == -1.0 && PyErr_Occurred() != nullptr) {
            if (PyErr_ExceptionMatches(PyExc_OverflowError) != 0) {
                PyErr_Clear();
                raiseOutOfRange(subject, object, code);
            }
            throw py::error_already_set();
        }
        return value;
    }
    if (isReal(object)) {
        const double value = PyFloat_AsDouble(object.ptr());
        throwIfPythonError();
        return value;
    }
    raiseTypeError(subject, code, object);
}

/// Rounds to the nearest float. A finite value that rounds to an infinity is refused; infinities and NaN pass.
float float32From(double value, py::handle object, const Subject& subject, std::string_view code) {
    const auto rounded = static_cast<float>(value);
    if (std::isfinite(value) && std::isinf(rounded)) {
        raiseOutOfRange(subject, object, code);
    }
    return rounded;
}

std::complex<double> complexFrom(py::handle object, const Subject& subject, std::string_view code) {
    if (PyComplex_Check(object.ptr())) {
        return {PyComplex_RealAsDouble(object.ptr()), PyComplex_ImagAsDouble(object.ptr())};
    }
    if (isInteger(object) || isReal(object)) {
        return {realFrom(object, subject, code), 0.0};
    }
    if (!PyBool_Check(object.ptr()) && PyObject_HasAttrString(object.ptr(), "__complex__") != 0) {
        const Py_complex value = PyComplex_AsCComplex(object.ptr());
        throwIfPythonError();
        return {value.real, value.imag};
    }
    raiseTypeError(subject, code, object);
}

/// Returns the Python scalar a zero-dimensional buffer holds, or the object itself when it holds none.
py::object scalarOfBuffer(py::handle object);

template<typename T>
constexpr Element elementFor() {
    if constexpr (std::same_as<T, bool>) {
        return Element::Bool;
    } else if constexpr (std::same_as<T, std::int8_t>) {
        return Element::Int8;
    } else if constexpr (std::same_as<T, std::int16_t>) {
        return Element::Int16;
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return Element::Int32;
    } else if constexpr (std::same_as<T, std::int64_t>) {
        return Element::Int64;
    } else if constexpr (std::same_as<T, std::uint8_t>) {
        return Element::UInt8;
    } else if constexpr (std::same_as<T, std::uint16_t>) {
        return Element::UInt16;
    } else if constexpr (std::same_as<T, std::uint32_t>) {
        return Element::UInt32;
    } else if constexpr (std::same_as<T, std::uint64_t>) {
        return Element::UInt64;
    } else if constexpr (std::same_as<T, float>) {
        return Element::Float32;
    } else if constexpr (std::same_as<T, double>) {
        return Element::Float64;
    } else if constexpr (std::same_as<T, std::complex<float>>) {
        return Element::ComplexFloat32;
    } else if constexpr (std::same_as<T, std::complex<double>>) {
        return Element::ComplexFloat64;
    } else {
        static_assert(std::same_as<T, std::pmr::string>);
        return Element::String;
    }
}

/// Converts a Python object to exactly `T`: a value out of range, or a fraction for an integer type, is an
/// `OverflowError`, and an object of the wrong kind a `TypeError`. `bool` is a kind of its own, apart from the
/// integers.
template<typename T>
T scalarFrom(py::handle rawObject, const Subject& subject) {
    const py::object           object = scalarOfBuffer(rawObject);
    constexpr std::string_view code   = nameOf(elementFor<T>());
    if constexpr (std::same_as<T, bool>) {
        if (!PyBool_Check(object.ptr())) {
            raiseTypeError(subject, code, object);
        }
        return object.ptr() == Py_True;
    } else if constexpr (std::same_as<T, std::pmr::string>) {
        return std::pmr::string(stringFrom(object, subject));
    } else if constexpr (std::integral<T>) {
        return integerFrom<T>(object, subject, code);
    } else if constexpr (std::same_as<T, float>) {
        return float32From(realFrom(object, subject, code), object, subject, code);
    } else if constexpr (std::same_as<T, double>) {
        return realFrom(object, subject, code);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
        const std::complex<double> value = complexFrom(object, subject, code);
        return {float32From(value.real(), object, subject, code), float32From(value.imag(), object, subject, code)};
    } else {
        static_assert(std::same_as<T, std::complex<double>>);
        return complexFrom(object, subject, code);
    }
}

Value valueFrom(py::handle object, const Subject& subject);

gr::pmt::Value::Map mapFrom(py::handle object, const Subject& subject) {
    Value::Map map;
    for (const auto& [key, item] : py::reinterpret_borrow<py::dict>(object)) {
        if (!PyUnicode_Check(key.ptr())) {
            throw py::type_error(std::format("{} has a key of type {}; map keys are str", subject.describe(), typeNameOf(key)));
        }
        const std::string name = stringFrom(key, subject);
        map.insert_or_assign(std::pmr::string(name), valueFrom(item, Subject{.key = name}));
    }
    return map;
}

/// A Python int as Int64, or as UInt64 above the Int64 range.
Value integerValueFrom(py::handle object, const Subject& subject) {
    const py::int_  number   = indexOf(object);
    int             overflow = 0;
    const long long value    = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
    throwIfPythonError();
    if (overflow == 0) {
        return Value(static_cast<std::int64_t>(value));
    }
    if (overflow > 0) {
        const unsigned long long unsignedValue = PyLong_AsUnsignedLongLong(number.ptr());
        if (PyErr_Occurred() == nullptr) {
            return Value(static_cast<std::uint64_t>(unsignedValue));
        }
        PyErr_Clear();
    }
    raiseOutOfRange(subject, object, "int64 and uint64");
}

/// The element a buffer format names, for the native byte order; `std::nullopt` for any other format.
std::optional<Element> elementOfBuffer(const py::buffer_info& buffer) {
    std::string_view format = buffer.format;
    if (!format.empty() && std::string_view("@=<>!").contains(format.front())) {
        const char order = format.front();
        format.remove_prefix(1UZ);
        const bool native = order == '@' || order == '=' || ((order == '<') == (std::endian::native == std::endian::little));
        if (!native) {
            return std::nullopt;
        }
    }
    const auto bySize = [&buffer](std::initializer_list<std::pair<py::ssize_t, Element>> sizes) -> std::optional<Element> {
        const auto found = std::ranges::find(sizes, buffer.itemsize, &std::pair<py::ssize_t, Element>::first);
        return found == sizes.end() ? std::nullopt : std::optional(found->second);
    };
    if (format == "?") {
        return bySize({{1, Element::Bool}});
    }
    if (format.size() == 1UZ && std::string_view("bhilqn").contains(format.front())) {
        return bySize({{1, Element::Int8}, {2, Element::Int16}, {4, Element::Int32}, {8, Element::Int64}});
    }
    if (format.size() == 1UZ && std::string_view("BHILQN").contains(format.front())) {
        return bySize({{1, Element::UInt8}, {2, Element::UInt16}, {4, Element::UInt32}, {8, Element::UInt64}});
    }
    if (format == "f" || format == "d") {
        return bySize({{4, Element::Float32}, {8, Element::Float64}});
    }
    if (format == "Zf" || format == "Zd" || format == "F" || format == "D") {
        return bySize({{8, Element::ComplexFloat32}, {16, Element::ComplexFloat64}});
    }
    return std::nullopt;
}

template<typename T>
T readElement(const std::byte* data) {
    if constexpr (std::same_as<T, bool>) {
        return std::to_integer<std::uint8_t>(*data) != 0U;
    } else {
        T value;
        std::memcpy(&value, data, sizeof(T));
        return value;
    }
}

/// Visits every element of a buffer in row-major order, following its strides.
template<typename Visitor>
void forEachBufferElement(const py::buffer_info& buffer, Visitor&& visit) {
    const auto               rank  = static_cast<std::size_t>(buffer.ndim);
    const std::size_t        count = static_cast<std::size_t>(std::ranges::fold_left(buffer.shape, py::ssize_t{1}, std::multiplies{}));
    std::vector<py::ssize_t> index(rank, 0);
    for (std::size_t n = 0UZ; n < count; ++n) {
        py::ssize_t offset = 0;
        for (std::size_t d = 0UZ; d < rank; ++d) {
            offset += index[d] * buffer.strides[d];
        }
        visit(static_cast<const std::byte*>(buffer.ptr) + offset);
        for (std::size_t d = rank; d-- > 0UZ;) {
            if (++index[d] < buffer.shape[d]) {
                break;
            }
            index[d] = 0;
        }
    }
}

template<typename T>
py::object scalarToPython(const T& value) {
    if constexpr (std::same_as<T, bool>) {
        return py::bool_(value);
    } else if constexpr (std::integral<T>) {
        return py::int_(value);
    } else if constexpr (std::floating_point<T>) {
        return py::float_(static_cast<double>(value));
    } else if constexpr (std::same_as<T, std::complex<float>> || std::same_as<T, std::complex<double>>) {
        return py::reinterpret_steal<py::object>(PyComplex_FromDoubles(static_cast<double>(value.real()), static_cast<double>(value.imag())));
    } else {
        static_assert(std::same_as<T, std::pmr::string>);
        return stringToPython(value);
    }
}

py::object scalarOfBuffer(py::handle object) {
    PyObject* raw = object.ptr();
    if (PyBool_Check(raw) || PyLong_Check(raw) || PyFloat_Check(raw) || PyComplex_Check(raw) || PyUnicode_Check(raw) || PyObject_CheckBuffer(object.ptr()) == 0) {
        return py::reinterpret_borrow<py::object>(object);
    }
    const py::buffer_info        buffer  = py::reinterpret_borrow<py::buffer>(object).request();
    const std::optional<Element> element = elementOfBuffer(buffer);
    if (buffer.ndim != 0 || !element.has_value()) {
        return py::reinterpret_borrow<py::object>(object);
    }
    return visitElementType(*element, [&buffer]<typename T>(std::type_identity<T>) -> py::object {
        if constexpr (std::same_as<T, std::pmr::string> || std::same_as<T, Value>) {
            std::unreachable();
        } else {
            return scalarToPython(readElement<T>(static_cast<const std::byte*>(buffer.ptr)));
        }
    });
}

/// A buffer as the typed tensor of its format, with the buffer's shape as its extents, or as a scalar when it has no
/// dimension.
Value valueFromBuffer(py::handle object, const Subject& subject) {
    const py::buffer_info        buffer  = py::reinterpret_borrow<py::buffer>(object).request();
    const std::optional<Element> element = elementOfBuffer(buffer);
    if (!element.has_value()) {
        throw py::type_error(std::format("{} is a buffer of format '{}' and item size {}, which no value type holds", subject.describe(), buffer.format, buffer.itemsize));
    }
    return visitElementType(*element, [&buffer]<typename T>(std::type_identity<T>) -> Value {
        if constexpr (std::same_as<T, std::pmr::string> || std::same_as<T, Value>) {
            std::unreachable();
        } else {
            if (buffer.ndim == 0) {
                return Value(readElement<T>(static_cast<const std::byte*>(buffer.ptr)));
            }
            const std::vector<std::size_t> extents(buffer.shape.begin(), buffer.shape.end());
            gr::Tensor<T>                  tensor(gr::extents_from, extents);
            auto                           out = tensor.begin();
            forEachBufferElement(buffer, [&out](const std::byte* data) { *out++ = readElement<T>(data); });
            return Value(std::move(tensor));
        }
    });
}

/// Converts a Python object to the value its own type selects: `bool` to Bool, `int` to Int64 (UInt64 above the
/// Int64 range), `float` to Float64, `complex` to complex<float64>, `str` to String, `None` to the empty value, a
/// `dict` with `str` keys to a map, a list or tuple to a tensor of type-erased elements, and a buffer to the typed
/// tensor of its format.
Value valueFrom(py::handle object, const Subject& subject) {
    const RecursionGuard guard;
    PyObject*            raw = object.ptr();
    if (raw == Py_None) {
        return Value{};
    }
    if (PyBool_Check(raw)) {
        return Value(raw == Py_True);
    }
    if (PyLong_Check(raw)) {
        return integerValueFrom(object, subject);
    }
    if (PyFloat_Check(raw)) {
        return Value(PyFloat_AS_DOUBLE(raw));
    }
    if (PyComplex_Check(raw)) {
        return Value(std::complex<double>(PyComplex_RealAsDouble(raw), PyComplex_ImagAsDouble(raw)));
    }
    if (PyUnicode_Check(raw)) {
        return Value(stringFrom(object, subject));
    }
    if (PyDict_Check(raw)) {
        return Value(mapFrom(object, subject));
    }
    if (isSequence(object)) {
        std::vector<Value> elements;
        elements.reserve(sizeOfSequence(object));
        for (std::size_t i = 0UZ; i < sizeOfSequence(object); ++i) {
            elements.push_back(valueFrom(itemOfSequence(object, i), subject.element(i)));
        }
        return Value(gr::Tensor<Value>(gr::data_from, elements));
    }
    if (PyObject_CheckBuffer(raw) != 0) {
        return valueFromBuffer(object, subject);
    }
    if (PyIndex_Check(raw)) {
        return integerValueFrom(object, subject);
    }
    throw py::type_error(std::format("{} of type {} converts to no value type", subject.describe(), typeNameOf(object)));
}

/// The extents of a nested list or tuple, read along its first elements, and its leaves in row-major order.
struct NestedSequence {
    std::vector<std::size_t> extents;
    std::vector<py::object>  leaves;
};

/// Appends the extents of a nested list or tuple, read along its first elements.
void appendExtents(py::handle level, std::vector<std::size_t>& extents) {
    if (!isSequence(level)) {
        return;
    }
    const RecursionGuard guard;
    extents.push_back(sizeOfSequence(level));
    if (extents.back() > 0UZ) {
        appendExtents(itemOfSequence(level, 0UZ), extents);
    }
}

NestedSequence flattenSequence(py::handle object, const Subject& subject) {
    NestedSequence nested;
    appendExtents(object, nested.extents);
    const std::function<void(py::handle, std::size_t)> collect = [&](py::handle level, std::size_t depth) {
        const RecursionGuard guard;
        if (depth == nested.extents.size()) {
            nested.leaves.push_back(py::reinterpret_borrow<py::object>(level));
            return;
        }
        if (!isSequence(level) || sizeOfSequence(level) != nested.extents[depth]) {
            throw py::type_error(std::format("{} is a ragged nested sequence", subject.describe()));
        }
        for (std::size_t i = 0UZ; i < nested.extents[depth]; ++i) {
            collect(itemOfSequence(level, i), depth + 1UZ);
        }
    };
    collect(object, 0UZ);
    return nested;
}

/// A tensor of exactly `element`, from a list or tuple (nested for a rank above one) or from a buffer, each element
/// converted by the scalar rule. A tensor of strings holds type-erased string elements.
Value tensorAs(py::handle object, Element element, const Subject& subject) {
    const std::string needed = std::format("tensor<{}>", nameOf(element));
    if (element == Element::Value) {
        if (!isSequence(object)) {
            raiseTypeError(subject, needed, object);
        }
        return valueFrom(object, subject);
    }

    NestedSequence nested;
    if (isSequence(object)) {
        nested = flattenSequence(object, subject);
    } else if (PyObject_CheckBuffer(object.ptr()) != 0 && !PyUnicode_Check(object.ptr())) {
        const py::buffer_info        buffer   = py::reinterpret_borrow<py::buffer>(object).request();
        const std::optional<Element> itemType = elementOfBuffer(buffer);
        if (buffer.ndim == 0 || !itemType.has_value()) {
            raiseTypeError(subject, needed, object);
        }
        nested.extents.assign(buffer.shape.begin(), buffer.shape.end());
        forEachBufferElement(buffer, [&nested, itemType](const std::byte* data) {
            nested.leaves.push_back(visitElementType(*itemType, [data]<typename T>(std::type_identity<T>) -> py::object {
                if constexpr (std::same_as<T, std::pmr::string> || std::same_as<T, Value>) {
                    std::unreachable();
                } else {
                    return scalarToPython(readElement<T>(data));
                }
            }));
        });
    } else {
        raiseTypeError(subject, needed, object);
    }

    return visitElementType(element, [&nested, &subject]<typename T>(std::type_identity<T>) -> Value {
        using Stored = std::conditional_t<std::same_as<T, std::pmr::string>, Value, T>;
        gr::Tensor<Stored> tensor(gr::extents_from, nested.extents);
        auto               out = tensor.begin();
        for (std::size_t i = 0UZ; i < nested.leaves.size(); ++i, ++out) {
            if constexpr (std::same_as<T, Value>) {
                std::unreachable();
            } else if constexpr (std::same_as<T, std::pmr::string>) {
                *out = Value(scalarFrom<std::pmr::string>(nested.leaves[i], subject.element(i)));
            } else {
                *out = scalarFrom<T>(nested.leaves[i], subject.element(i));
            }
        }
        return Value(std::move(tensor));
    });
}

/// Converts a Python object to a value of exactly the type `code` names.
Value valueAs(py::handle object, const TypeCode& code, const Subject& subject) {
    switch (code.kind) {
    case TypeCode::Kind::None:
        if (!object.is_none()) {
            raiseTypeError(subject, "none", object);
        }
        return Value{};
    case TypeCode::Kind::Map:
        if (!PyDict_Check(object.ptr())) {
            raiseTypeError(subject, "map", object);
        }
        return Value(mapFrom(object, subject));
    case TypeCode::Kind::Scalar:
        return visitElementType(code.element, [&]<typename T>(std::type_identity<T>) -> Value {
            if constexpr (std::same_as<T, Value>) {
                std::unreachable();
            } else {
                return Value(scalarFrom<T>(object, subject));
            }
        });
    case TypeCode::Kind::Tensor: return tensorAs(object, code.element, subject);
    }
    std::unreachable();
}

Value convertFrom(py::handle object, const std::optional<std::string>& typeCode) { return typeCode.has_value() ? valueAs(object, parseTypeCode(*typeCode), Subject{}) : valueFrom(object, Subject{}); }

py::object toPython(const Value& value);

py::dict mapToPython(const Value::Map& map) {
    py::dict result;
    for (const auto& [key, item] : map) {
        result[stringToPython(key)] = toPython(item);
    }
    return result;
}

/// The row-major elements of a tensor, nested into lists by its extents when its rank is above one.
py::list nestByExtents(const py::list& flat, std::span<const std::size_t> extents, std::size_t offset) {
    if (extents.size() <= 1UZ) {
        const std::size_t count = extents.empty() ? flat.size() : extents.front();
        py::list          level(count);
        for (std::size_t i = 0UZ; i < count; ++i) {
            level[i] = py::object(flat[offset + i]);
        }
        return level;
    }
    const std::size_t stride = std::ranges::fold_left(extents.subspan(1UZ), 1UZ, std::multiplies{});
    py::list          level(extents.front());
    for (std::size_t i = 0UZ; i < extents.front(); ++i) {
        level[i] = nestByExtents(flat, extents.subspan(1UZ), offset + i * stride);
    }
    return level;
}

template<typename T>
py::list tensorToPython(const gr::Tensor<T>& tensor) {
    py::list    flat(tensor.size());
    std::size_t i = 0UZ;
    for (const auto& element : tensor) {
        if constexpr (std::same_as<T, Value>) {
            flat[i++] = toPython(element);
        } else {
            flat[i++] = scalarToPython(static_cast<const T&>(element));
        }
    }
    return nestByExtents(flat, tensor.extents(), 0UZ);
}

/// Converts a value to Python: scalars to Python scalars, a tensor to a list (nested when its rank is above one), a
/// map to a `dict` and the empty value to `None`.
py::object toPython(const Value& value) {
    const RecursionGuard guard;
    if (value.is_map()) {
        return mapToPython(*value.get_if<Value::Map>());
    }
    if (value.is_monostate()) {
        return py::none();
    }
    return visitElementType(elementOf(value.value_type()), [&value]<typename T>(std::type_identity<T>) -> py::object {
        if (value.is_tensor()) {
            const auto* tensor = value.get_if<gr::Tensor<T>>();
            return tensor != nullptr ? py::object(tensorToPython(*tensor)) : py::object(py::list());
        }
        if constexpr (std::same_as<T, Value>) {
            std::unreachable();
        } else if constexpr (std::same_as<T, std::pmr::string>) {
            return stringToPython(value.value_or(std::string_view{}));
        } else {
            return scalarToPython(*value.get_if<T>());
        }
    });
}

/// Converts a settings map from Python. `parameters` is a dict with str keys, or None for an empty map; `types` maps
/// a key to the type code its value converts to, and a key it does not name takes the type its Python value selects.
gr::property_map propertyMapFrom(const py::object& parameters, const py::object& types) {
    gr::property_map map;
    if (parameters.is_none()) {
        return map;
    }
    if (!PyDict_Check(parameters.ptr())) {
        throw py::type_error(std::format("parameters need a dict, got {}", typeNameOf(parameters)));
    }
    if (!types.is_none() && !PyDict_Check(types.ptr())) {
        throw py::type_error(std::format("types need a dict of type codes, got {}", typeNameOf(types)));
    }
    for (const auto& [key, item] : py::reinterpret_borrow<py::dict>(parameters)) {
        if (!PyUnicode_Check(key.ptr())) {
            throw py::type_error(std::format("parameter keys are str, got {}", typeNameOf(key)));
        }
        const std::string name = stringFrom(key, Subject{});
        const Subject     subject{.key = name};
        PyObject*         typeCode = types.is_none() ? nullptr : PyDict_GetItemWithError(types.ptr(), key.ptr());
        throwIfPythonError();
        Value value = typeCode == nullptr ? valueFrom(item, subject) : valueAs(item, parseTypeCode(stringFrom(typeCode, subject)), subject);
        map.insert_or_assign(std::pmr::string(name), std::move(value));
    }
    return map;
}

/// The class `gr4._gr4.Error`, created when the module initializes and kept for the life of the process.
PyObject* errorClass = nullptr;

/// Sets `Error(message, where, time_ns)` as the pending Python exception.
void setError(std::string_view message, std::string_view where, std::uint64_t timeNs) {
    const py::object error = py::reinterpret_borrow<py::object>(errorClass)(stringToPython(message), stringToPython(where), timeNs);
    PyErr_SetObject(errorClass, error.ptr());
}

[[noreturn]] void raiseError(std::string_view message, std::string_view where, std::uint64_t timeNs = 0U) {
    setError(message, where, timeNs);
    throw py::error_already_set();
}

[[noreturn]] void raiseError(const gr::RuntimeError& error) { raiseError(error.message, error.where, error.time); }

template<typename T>
T valueOrRaise(std::expected<T, gr::RuntimeError>&& result) {
    if (!result.has_value()) {
        raiseError(result.error());
    }
    return std::move(*result);
}

void raiseIfFailed(const std::expected<void, gr::RuntimeError>& result) {
    if (!result.has_value()) {
        raiseError(result.error());
    }
}

std::uint64_t nanosecondsOf(std::chrono::system_clock::time_point when) { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(when.time_since_epoch()).count()); }

/// Raises a C++ exception a call lets out as `Error`. The exceptions of pybind11 itself keep their Python types.
void translateException(std::exception_ptr exception) {
    try {
        std::rethrow_exception(exception);
    } catch (const py::builtin_exception&) {
        throw;
    } catch (const gr::exception& error) {
        setError(error.message, std::format("{}:{}", error.sourceLocation.file_name(), error.sourceLocation.line()), nanosecondsOf(error.errorTime));
    } catch (const std::exception& error) {
        setError(error.what(), "", 0U);
    } catch (...) {
        setError("a C++ exception of unknown type", "", 0U);
    }
}

void defineErrorClass(py::module_& module) {
    auto error = py::reinterpret_steal<py::object>(PyErr_NewExceptionWithDoc("gr4._gr4.Error",
        "A failure the runtime reported.\n\n"
        "message is the reason, where the framework's source location or the failing method, and time_ns the time the "
        "framework reported the error, in nanoseconds since the epoch, or 0.",
        PyExc_Exception, nullptr));
    if (!error) {
        throw py::error_already_set();
    }
    error.attr("__init__") = py::cpp_function(
        [](const py::object& self, const py::object& message, const py::object& where, const py::object& timeNs) {
            self.attr("args")    = py::make_tuple(message);
            self.attr("message") = message;
            self.attr("where")   = where;
            self.attr("time_ns") = timeNs;
        },
        py::is_method(error), py::arg("message") = "", py::arg("where") = "", py::arg("time_ns") = 0);
    module.attr("Error") = error;
    errorClass           = error.release().ptr();
}

/// The value of GNURADIO4_PLUGIN_DIRECTORIES when a call first reached the plugin loader, which reads it then.
std::optional<std::string> loaderDirectories;

void noteLoaderUse() {
    if (!loaderDirectories.has_value()) {
        const char* value = std::getenv(kPluginDirectoriesVariable);
        loaderDirectories = value != nullptr ? value : "";
    }
}

/// The directories the loader reads, split as the loader splits the variable: at each colon, empty entries dropped.
std::vector<std::string> pluginDirectories() {
    const char*              current = std::getenv(kPluginDirectoriesVariable);
    const std::string_view   value   = loaderDirectories.has_value() ? std::string_view(*loaderDirectories) : std::string_view(current != nullptr ? current : "");
    std::vector<std::string> directories;
    for (const auto part : std::views::split(value, ':')) {
        if (!std::ranges::empty(part)) {
            directories.emplace_back(std::ranges::begin(part), std::ranges::end(part));
        }
    }
    return directories;
}

/// Runs `call` with the GIL released. Python objects are converted before and after, never inside.
template<typename Call>
std::invoke_result_t<Call> withoutGil(Call&& call) {
    py::gil_scoped_release release;
    return std::forward<Call>(call)();
}

bool interpreterIsFinalizing() {
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsFinalizing() != 0;
#else
    return _Py_IsFinalizing() != 0;
#endif
}

/// Runs `call` without the GIL when this thread holds it, for a destructor, which may run with or without it. While
/// the interpreter finalizes the GIL stays held; the calls made here take no Python object.
template<typename Call>
void withoutGilWhenHeld(Call&& call) {
    if (PyGILState_Check() != 0 && !interpreterIsFinalizing()) {
        py::gil_scoped_release release;
        std::forward<Call>(call)();
    } else {
        std::forward<Call>(call)();
    }
}

/// Deletes an object without the GIL. Destroying a block can block, and destroying a scheduler waits for its
/// watchdog.
template<typename T>
struct DeleteWithoutGil {
    void operator()(T* object) const noexcept {
        withoutGilWhenHeld([object] { delete object; });
    }
};

/// The holder of a bound object whose destruction can destroy blocks or a scheduler.
template<typename T>
using GilFreeHolder = std::unique_ptr<T, DeleteWithoutGil<T>>;

template<typename T>
GilFreeHolder<T> holdWithoutGil(T object) {
    return GilFreeHolder<T>(new T(std::move(object)));
}

std::string blockRepr(const gr::BlockHandle& block) {
    if (!block.valid()) {
        return "<BlockHandle invalid>";
    }
    return std::format("<BlockHandle name='{}' unique_name='{}' type='{}'>", block.name(), block.uniqueName(), block.typeName());
}

/// A Python `RuntimeGraph`: an owning graph or a view of one, and empty once `Runtime.create` consumed it.
struct GraphObject {
    std::optional<gr::RuntimeGraph> graph;
    bool                            isView = false;

    [[nodiscard]] gr::RuntimeGraph& get() {
        if (!graph.has_value()) {
            raiseError("the graph moved into a scheduler; Runtime.graph() views it there", "RuntimeGraph");
        }
        return *graph;
    }
};

gr::RuntimeGraph::Recursive recursion(bool recursive) { return recursive ? gr::RuntimeGraph::Recursive::Yes : gr::RuntimeGraph::Recursive::No; }

void defineBlockHandle(py::module_& module) {
    py::class_<gr::BlockHandle, GilFreeHolder<gr::BlockHandle>>(module, "BlockHandle", "One block of a graph: its names, its settings and its meta-information. Two handles are equal when they name the same block.")
        .def_property_readonly("valid", &gr::BlockHandle::valid, "True when the handle names a block.")
        .def_property_readonly(
            "name", [](const gr::BlockHandle& block) { return std::string(block.name()); }, "The name the block was given.")
        .def_property_readonly(
            "unique_name", [](const gr::BlockHandle& block) { return std::string(block.uniqueName()); }, "The name that identifies the block in its process.")
        .def_property_readonly(
            "type_name", [](const gr::BlockHandle& block) { return std::string(block.typeName()); }, "The registry key of the block's type.")
        .def(
            "get", [](const gr::BlockHandle& block, const std::optional<std::vector<std::string>>& keys) { return mapToPython(keys.has_value() ? block.get(std::span<const std::string>(*keys)) : block.get()); }, py::arg("keys") = py::none(), "Returns the active settings, all of them or those keys names.")
        .def(
            "get_value",
            [](const gr::BlockHandle& block, const std::string& key) {
                const std::optional<Value> value = block.get(key);
                if (!value.has_value()) {
                    throw py::key_error(key);
                }
                return toPython(*value);
            },
            py::arg("key"), "Returns the active value of one setting. KeyError when the block has no setting of that name.")
        .def(
            "staged_parameters", [](const gr::BlockHandle& block) { return mapToPython(block.stagedParameters()); }, "Returns the settings staged for the block's next work call.")
        .def(
            "default_parameters", [](const gr::BlockHandle& block) { return mapToPython(block.defaultParameters()); }, "Returns the block's default settings.")
        .def(
            "setting_types",
            [](const gr::BlockHandle& block) {
                py::dict types;
                for (const auto& [key, value] : block.get()) {
                    types[stringToPython(key)] = typeCodeOf(value);
                }
                return types;
            },
            "Returns the type code of every active setting, by key.")
        .def(
            "set_staged", [](gr::BlockHandle& block, const py::object& parameters, const py::object& types) { return mapToPython(valueOrRaise(block.setStaged(propertyMapFrom(parameters, types)))); }, py::arg("parameters"), py::arg("types") = py::none(), "Stages settings for the block's next work call; types maps a key to its type code. Returns the settings the block does not declare. A value that does not fit its type code raises TypeError or OverflowError; a value the block cannot convert or refuses raises Error. Either way nothing is staged.")
        .def(
            "meta_information", [](const gr::BlockHandle& block) { return mapToPython(block.metaInformation()); }, "Returns the block's meta-information.")
        .def(
            "set_meta_information", [](gr::BlockHandle& block, const py::object& information) { block.setMetaInformation(propertyMapFrom(information, py::none())); }, py::arg("information"), "Replaces the block's meta-information.")
        .def(
            "__eq__", [](const gr::BlockHandle& self, const gr::BlockHandle& other) { return self.uniqueName() == other.uniqueName(); }, py::is_operator())
        .def("__hash__", [](const gr::BlockHandle& block) { return py::hash(py::str(std::string(block.uniqueName()))); })
        .def("__repr__", &blockRepr);
}

void definePortAndEdge(py::module_& module) {
    py::class_<gr::RuntimePort>(module, "RuntimePort", "One port of a block, as RuntimeGraph.input_ports() and output_ports() describe it.").def_readonly("name", &gr::RuntimePort::name, "The name connect() accepts: 'out', or 'in#1' for element 1 of a collection.").def_readonly("type_name", &gr::RuntimePort::typeName, "The value type, as the port spells it.").def_readonly("is_input", &gr::RuntimePort::isInput, "True for an input port.").def_readonly("is_message", &gr::RuntimePort::isMessage, "True for a message port, False for a stream port.").def_readonly("is_optional", &gr::RuntimePort::isOptional, "True when the port may stay unconnected.").def_readonly("is_synchronous", &gr::RuntimePort::isSynchronous, "True when the port moves samples in step with the block's other synchronous ports.").def_readonly("is_connected", &gr::RuntimePort::isConnected, "True while a running scheduler holds the port's edge connected.").def_readonly("domain", &gr::RuntimePort::domain, "The compute domain of the port.").def_readonly("collection", &gr::RuntimePort::collection, "The name of the collection the port belongs to, or ''.").def_readonly("index", &gr::RuntimePort::index, "The port's position in its collection, or 0.").def_readonly("min_samples", &gr::RuntimePort::minSamples, "The fewest samples the port takes per work call.").def_readonly("max_samples", &gr::RuntimePort::maxSamples, "The most samples the port takes per work call.").def("__repr__", [](const gr::RuntimePort& port) { return std::format("<RuntimePort {} {} {}>", port.name, port.isInput ? "input" : "output", port.typeName); });

    py::class_<gr::RuntimeEdge, GilFreeHolder<gr::RuntimeEdge>>(module, "RuntimeEdge", "One edge of a graph, as RuntimeGraph.edges() lists it.").def_property_readonly("source_block", [](const gr::RuntimeEdge& edge) { return edge.sourceBlock; }, "The block the edge leaves.").def_readonly("source_port", &gr::RuntimeEdge::sourcePort, "The output port the edge leaves.").def_property_readonly("destination_block", [](const gr::RuntimeEdge& edge) { return edge.destinationBlock; }, "The block the edge enters.").def_readonly("destination_port", &gr::RuntimeEdge::destinationPort, "The input port the edge enters.").def_property_readonly("min_buffer_size", [](const gr::RuntimeEdge& edge) { return edge.edge.minBufferSize; }, "The smallest buffer the edge asks for, the framework's default when none was given.").def_property_readonly("weight", [](const gr::RuntimeEdge& edge) { return edge.edge.weight; }, "The edge's weight.").def_property_readonly("name", [](const gr::RuntimeEdge& edge) { return edge.edge.name; }, "The edge's name.").def("__repr__", [](const gr::RuntimeEdge& edge) { return std::format("<RuntimeEdge {}.{} -> {}.{}>", edge.sourceBlock.uniqueName(), edge.sourcePort, edge.destinationBlock.uniqueName(), edge.destinationPort); });
}

void defineRuntimeGraph(py::module_& module) {
    py::class_<GraphObject, GilFreeHolder<GraphObject>>(module, "RuntimeGraph",
        "A flow graph built by registry name.\n\n"
        "A RuntimeGraph owns its graph, or views one a subgraph block or a scheduler owns. A graph handed to "
        "Runtime.create moves into the scheduler, and every later call on this object raises Error.\n\n"
        "The constructor, emplace(), emplace_subgraph(), remove() and from_yaml() release the GIL, and so "
        "does the deallocation of a graph, a block handle or an edge.")
        .def(py::init([] {
            noteLoaderUse();
            return holdWithoutGil(GraphObject{.graph = withoutGil([] { return gr::RuntimeGraph(); })});
        }),
            "Creates an empty graph.")
        .def(
            "emplace",
            [](GraphObject& self, const std::string& type, const std::string& name, const py::object& parameters, const py::object& types) {
                gr::property_map  initialParameters = propertyMapFrom(parameters, types);
                gr::RuntimeGraph& graph             = self.get();
                return valueOrRaise(withoutGil([&] { return graph.emplace(type, name, std::move(initialParameters)); }));
            },
            py::arg("type"), py::arg("name"), py::arg("parameters") = py::none(), py::arg("types") = py::none(), "Instantiates the block type registered as type, names it and adds it. The parameters are in force when the block starts; types maps a key to its type code. A parameter the block does not declare or refuses raises Error with the block's message.")
        .def(
            "emplace_subgraph",
            [](GraphObject& self, const std::string& name, const py::object& parameters) {
                gr::property_map  initialParameters = propertyMapFrom(parameters, py::none());
                gr::RuntimeGraph& graph             = self.get();
                return valueOrRaise(withoutGil([&] { return graph.emplaceSubgraph(name, std::move(initialParameters)); }));
            },
            py::arg("name"), py::arg("parameters") = py::none(), "Adds a nested graph; interior() opens it for wiring.")
        .def(
            "interior", [](GraphObject& self, const gr::BlockHandle& block) { return holdWithoutGil(GraphObject{.graph = valueOrRaise(self.get().interior(block)), .isView = true}); }, py::arg("block"), "Returns a view of a subgraph's interior, or of a nested scheduler's graph.")
        .def(
            "blocks", [](GraphObject& self, bool recursive) { return self.get().blocks(recursion(recursive)); }, py::arg("recursive") = false, "Returns the blocks of this level of the graph, or of every level.")
        .def(
            "find", [](GraphObject& self, const std::string& uniqueName, bool recursive) { return valueOrRaise(self.get().find(uniqueName, recursion(recursive))); }, py::arg("unique_name"), py::arg("recursive") = true, "Returns the block with that unique name.")
        .def(
            "remove",
            [](GraphObject& self, const gr::BlockHandle& block) {
                gr::RuntimeGraph& graph = self.get();
                raiseIfFailed(withoutGil([&graph, &block] { return graph.remove(block); }));
            },
            py::arg("block"), "Removes a block and its edges.")
        .def(
            "connect",
            [](GraphObject& self, const gr::BlockHandle& source, const std::string& sourcePort, const gr::BlockHandle& destination, const std::string& destinationPort, std::size_t minBufferSize, std::int32_t weight, std::string name) { //
                raiseIfFailed(self.get().connect(source, sourcePort, destination, destinationPort, gr::EdgeSpec{.minBufferSize = minBufferSize, .weight = weight, .name = std::move(name)}));
            },
            py::arg("source"), py::arg("source_port"), py::arg("destination"), py::arg("destination_port"), py::arg("min_buffer_size") = 0U, py::arg("weight") = 0, py::arg("name") = "", "Connects an output port to an input port. A min_buffer_size of 0 selects the framework's default.")
        .def(
            "disconnect", [](GraphObject& self, const gr::BlockHandle& source, const std::string& sourcePort, const gr::BlockHandle& destination, const std::string& destinationPort) { raiseIfFailed(self.get().disconnect(source, sourcePort, destination, destinationPort)); }, py::arg("source"), py::arg("source_port"), py::arg("destination"), py::arg("destination_port"), "Removes the edge between two ports.")
        .def(
            "input_port_names", [](GraphObject& self, const gr::BlockHandle& block) { return self.get().inputPortNames(block); }, py::arg("block"), "Returns the names of a block's input ports.")
        .def(
            "output_port_names", [](GraphObject& self, const gr::BlockHandle& block) { return self.get().outputPortNames(block); }, py::arg("block"), "Returns the names of a block's output ports.")
        .def(
            "input_ports", [](GraphObject& self, const gr::BlockHandle& block) { return self.get().inputPorts(block); }, py::arg("block"), "Describes the ports input_port_names() lists, in the same order.")
        .def(
            "output_ports", [](GraphObject& self, const gr::BlockHandle& block) { return self.get().outputPorts(block); }, py::arg("block"), "Describes the ports output_port_names() lists, in the same order.")
        .def(
            "edges", [](GraphObject& self) { return self.get().edges(); }, "Returns the edges of this level of the graph.")
        .def(
            "export_port", [](GraphObject& self, const gr::BlockHandle& innerBlock, bool isInput, const std::string& port, const std::string& exportedName) { raiseIfFailed(self.get().exportPort(innerBlock, isInput, port, exportedName)); }, py::arg("inner_block"), py::arg("is_input"), py::arg("port"), py::arg("exported_name"), "Publishes a port of an interior block on the subgraph, under exported_name. Call on the view interior() returns.")
        .def(
            "unexport_port", [](GraphObject& self, const gr::BlockHandle& innerBlock, bool isInput, const std::string& port) { raiseIfFailed(self.get().unexportPort(innerBlock, isInput, port)); }, py::arg("inner_block"), py::arg("is_input"), py::arg("port"), "Withdraws a port export_port() published.")
        .def(
            "exported_input_ports", [](GraphObject& self) { return mapToPython(self.get().exportedInputPorts()); }, "Returns the exported input ports: block unique name to port name to {'exportedName': name}.")
        .def(
            "exported_output_ports", [](GraphObject& self) { return mapToPython(self.get().exportedOutputPorts()); }, "Returns the exported output ports: block unique name to port name to {'exportedName': name}.")
        .def_static(
            "from_yaml",
            [](const std::string& document) {
                noteLoaderUse();
                auto loaded = withoutGil([&document] { return gr::RuntimeGraph::fromYaml(document); });
                return holdWithoutGil(GraphObject{.graph = valueOrRaise(std::move(loaded))});
            },
            py::arg("document"), "Loads a graph document into a new graph. A document that does not parse raises Error naming the line and column.")
        .def(
            "to_yaml", [](GraphObject& self) { return valueOrRaise(self.get().toYaml()); }, "Writes the graph as a graph document.")
        .def("__repr__", [](const GraphObject& self) {
            if (!self.graph.has_value()) {
                return std::string("<RuntimeGraph moved into a scheduler>");
            }
            return std::format("<RuntimeGraph {} with {} blocks>", self.isView ? "view" : "owner", self.graph->blocks().size());
        });
}

/// The longest timeout `Runtime.wait()` sets a deadline for, in seconds; a longer one waits without a deadline. A
/// deadline further ahead overflows the clock.
constexpr double kLongestTimedWait = 1e9;

/// How long a blocking call waits without the GIL before it takes the GIL to run the signal handlers.
constexpr std::chrono::milliseconds kSignalCheckInterval{100};

using Clock = std::chrono::steady_clock;

/// Waits until no run of `held` is in progress or the deadline passes, and returns whether no run is in progress.
/// The GIL is released while the call waits and taken every kSignalCheckInterval to run the signal handlers. An
/// exception a handler raises, the KeyboardInterrupt of Ctrl-C among them, ends the wait. Called with the GIL held.
bool waitForRun(const gr::Runtime& held, std::optional<Clock::time_point> deadline) {
    for (;;) {
        Clock::duration slice = kSignalCheckInterval;
        if (deadline.has_value()) {
            slice = std::clamp<Clock::duration>(*deadline - Clock::now(), Clock::duration::zero(), slice);
        }
        if (withoutGil([&held, slice] { return held.waitFor(slice); })) {
            return true;
        }
        if (deadline.has_value() && Clock::now() >= *deadline) {
            return false;
        }
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    }
}

/// Stops the run of `held` in progress and returns once it has ended. The GIL is released while the call waits in
/// `Runtime::stopFor()` and taken every kSignalCheckInterval to run the signal handlers. An exception a handler
/// raises, the KeyboardInterrupt of Ctrl-C among them, ends the wait and leaves the stop requested and the run
/// ending. Called with the GIL held.
void stopRun(gr::Runtime& held) {
    while (!withoutGil([&held] { return held.stopFor(kSignalCheckInterval); })) {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    }
}

/// A Python `Runtime`:a scheduler holding a graph, and the last run's result once close() released the runtime.
///
/// `mutex` guards every member. A call that uses the runtime holds its own reference to it, so `close()` on another
/// thread destroys the runtime only after that call returns. The runtime's destructor stops the run in progress and
/// waits for its end. The deleter runs that destructor without the GIL. No thread takes the GIL while it holds
/// `mutex`.
struct RuntimeObject {
    std::mutex                            mutex;
    std::shared_ptr<gr::Runtime>          runtime;           // empty once closed
    std::expected<void, gr::RuntimeError> closedResult;      // the last run's result, kept when close() releases the runtime
    bool                                  closing   = false; // close() began, and no run starts
    std::size_t                           quiescing = 0UZ;   // the quiesce() guards held

    RuntimeObject()                                = default;
    RuntimeObject(const RuntimeObject&)            = delete;
    RuntimeObject& operator=(const RuntimeObject&) = delete;

    /// The runtime, held for one call. Raises Error once the runtime is closed.
    [[nodiscard]] std::shared_ptr<gr::Runtime> acquire() {
        std::shared_ptr<gr::Runtime> held;
        {
            std::lock_guard lock(mutex);
            held = runtime;
        }
        if (held == nullptr) {
            raiseError("the runtime is closed", "Runtime");
        }
        return held;
    }

    /// Starts a run on the runtime's thread and returns the runtime, held for the call. Raises Error naming `where`
    /// when no run starts: while a run is in progress, when the thread does not start, and once close() began. The
    /// runtime's start() runs with `mutex` held, and close() sets `closing` under `mutex`.
    std::shared_ptr<gr::Runtime> start(const char* where) {
        std::optional<gr::RuntimeError>    refusal;
        const std::shared_ptr<gr::Runtime> held = withoutGil([this, &refusal] {
            std::lock_guard lock(mutex);
            if (runtime == nullptr || closing) {
                return std::shared_ptr<gr::Runtime>();
            }
            refusal = runtime->start();
            return runtime;
        });
        if (held == nullptr) {
            raiseError("the runtime is closed", where);
        }
        if (refusal.has_value()) {
            raiseError(refusal->message, where, refusal->time);
        }
        return held;
    }

    /// Returns the result of the runtime's last run, or the result close() kept when it released the runtime.
    [[nodiscard]] std::expected<void, gr::RuntimeError> result() {
        std::shared_ptr<gr::Runtime> held;
        {
            std::lock_guard lock(mutex);
            if (runtime == nullptr) {
                return closedResult;
            }
            held = runtime;
        }
        return held->result();
    }
};

/// The context manager `Runtime.quiesce()` returns: it holds the runtime's quiescence guard between enter and exit.
struct QuiescenceObject {
    py::object                                  owner; // the Python Runtime, alive while the guard is held
    RuntimeObject*                              runtime = nullptr;
    std::shared_ptr<gr::Runtime>                held; // the runtime the guard quiesces
    std::optional<gr::Runtime::QuiescenceGuard> guard;

    QuiescenceObject(py::object ownerObject, RuntimeObject* runtimeObject) : owner(std::move(ownerObject)), runtime(runtimeObject) {}
    QuiescenceObject(const QuiescenceObject&)            = delete;
    QuiescenceObject& operator=(const QuiescenceObject&) = delete;

    ~QuiescenceObject() { release(); }

    void release() {
        if (guard.has_value()) {
            withoutGilWhenHeld([this] {
                guard.reset();
                held.reset();
            });
            std::lock_guard lock(runtime->mutex);
            --runtime->quiescing;
        }
    }
};

void defineEnumerations(py::module_& module) {
    py::native_enum<gr::Runtime::State>(module, "State", "enum.IntEnum", "The lifecycle state of a scheduler.").value("IDLE", gr::Runtime::State::Idle).value("INITIALIZED", gr::Runtime::State::Initialized).value("RUNNING", gr::Runtime::State::Running).value("REQUESTED_PAUSE", gr::Runtime::State::RequestedPause).value("PAUSED", gr::Runtime::State::Paused).value("REQUESTED_STOP", gr::Runtime::State::RequestedStop).value("STOPPED", gr::Runtime::State::Stopped).value("ERROR", gr::Runtime::State::Error).finalize();
    py::native_enum<gr::RuntimeCommand>(module, "Command", "enum.IntEnum", "A command of the message plane.").value("INVALID", gr::RuntimeCommand::Invalid).value("GET", gr::RuntimeCommand::Get).value("SET", gr::RuntimeCommand::Set).value("PARTIAL", gr::RuntimeCommand::Partial).value("FINAL", gr::RuntimeCommand::Final).value("READY", gr::RuntimeCommand::Ready).value("DISCONNECT", gr::RuntimeCommand::Disconnect).value("SUBSCRIBE", gr::RuntimeCommand::Subscribe).value("UNSUBSCRIBE", gr::RuntimeCommand::Unsubscribe).value("NOTIFY", gr::RuntimeCommand::Notify).value("HEARTBEAT", gr::RuntimeCommand::Heartbeat).finalize();
}

void defineRuntimeEvent(py::module_& module) {
    py::class_<gr::RuntimeEvent>(module, "RuntimeEvent", "One report the running graph made on the message plane.").def_readonly("source", &gr::RuntimeEvent::source, "The unique name of the reporting block.").def_readonly("endpoint", &gr::RuntimeEvent::endpoint, "The property endpoint it reported on.").def_readonly("command", &gr::RuntimeEvent::command, "FINAL for the reply to a request, NOTIFY for a notification.").def_readonly("client_request_id", &gr::RuntimeEvent::clientRequestID, "The id of the request or subscription the event answers, or ''.").def_readonly("is_error", &gr::RuntimeEvent::isError, "True when the event reports an error.").def_readonly("text", &gr::RuntimeEvent::text, "The rendered error text of an error event.").def_property_readonly("data", [](const gr::RuntimeEvent& event) { return mapToPython(event.data); }, "The message payload of an event that is not an error.").def_readonly("time_ns", &gr::RuntimeEvent::time, "The time the error was raised, in nanoseconds since the epoch, or 0.").def("__repr__", [](const gr::RuntimeEvent& event) { return std::format("<RuntimeEvent {} {} command={} id='{}'{}>", event.source, event.endpoint, static_cast<int>(event.command), event.clientRequestID, event.isError ? " error" : ""); });
}

void defineRuntime(py::module_& module) {
    py::class_<QuiescenceObject>(module, "Quiescence", "The context manager Runtime.quiesce() returns. The work loop stays quiesced from entry to exit.")
        .def(
            "__enter__",
            [](QuiescenceObject& self) -> QuiescenceObject& {
                if (self.guard.has_value()) {
                    raiseError("the quiesce() block is entered already", "Runtime.quiesce");
                }
                std::shared_ptr<gr::Runtime> held = self.runtime->acquire();
                {
                    std::lock_guard lock(self.runtime->mutex);
                    ++self.runtime->quiescing;
                }
                try {
                    self.guard.emplace(withoutGil([&held] { return held->quiesce(); }));
                } catch (...) {
                    std::lock_guard lock(self.runtime->mutex);
                    --self.runtime->quiescing;
                    throw;
                }
                self.held = std::move(held);
                return self;
            },
            py::return_value_policy::reference)
        .def("__exit__", [](QuiescenceObject& self, const py::args&) {
            self.release();
            return false;
        });

    py::class_<RuntimeObject>(module, "Runtime",
        "A scheduler holding a graph, and the subscriber on its message plane.\n\n"
        "The runtime runs the graph on a thread it owns, one run at a time, and keeps the last run's result: "
        "run_and_wait(), start() and run_in_thread() start a run, and each raises Error while a run is in progress. "
        "close() stops the run, waits for its end and releases the runtime; the destructor does the same.\n\n"
        "create(), run_and_wait(), start(), run_in_thread(), stop(), wait(), request_state(), close() and entering and "
        "leaving a quiesce() block release the GIL, and so does the deallocation of a runtime. run_and_wait(), "
        "wait(), stop() and close() run the signal handlers every 100 ms while they wait in the main thread, and "
        "Ctrl-C raises KeyboardInterrupt from them. A KeyboardInterrupt from stop() or close() leaves the stop "
        "requested and the run ending, and close() after it waits for the end.")
        .def_static(
            "create",
            [](GraphObject& graph, const std::optional<std::string>& type, const py::object& settings, const py::object& types) {
                gr::property_map schedulerParameters = propertyMapFrom(settings, types);
                if (graph.isView) {
                    raiseError("Runtime.create takes a graph the RuntimeGraph owns, not a view", "Runtime.create");
                }
                std::ignore                     = graph.get();
                const std::string schedulerType = type.value_or(std::string(gr::Runtime::kDefaultScheduler));
                noteLoaderUse();
                const std::vector<std::string> schedulerTypes = withoutGil([] { return gr::RuntimeGraph::availableSchedulerTypes(); });
                if (std::ranges::find(schedulerTypes, schedulerType) == schedulerTypes.end()) {
                    raiseError(std::format("unknown scheduler type '{}'; available_scheduler_types() lists the {} types there are", schedulerType, schedulerTypes.size()), "Runtime.create");
                }
                gr::RuntimeGraph consumed = std::move(*graph.graph);
                graph.graph.reset();
                auto        created = withoutGil([&] { return gr::Runtime::create(std::move(consumed), schedulerType, std::move(schedulerParameters)); });
                gr::Runtime runtime = valueOrRaise(std::move(created));
                auto        object  = std::make_unique<RuntimeObject>();
                object->runtime     = std::shared_ptr<gr::Runtime>(new gr::Runtime(std::move(runtime)), DeleteWithoutGil<gr::Runtime>{});
                return object;
            },
            py::arg("graph"), py::arg("type") = py::none(), py::arg("settings") = py::none(), py::arg("types") = py::none(),
            "Instantiates the scheduler registered as type, the runtime's default when type is None, with the given "
            "settings, and hands it the graph, which moves into the scheduler.")
        .def(
            "run_and_wait",
            [](RuntimeObject& self) {
                const std::shared_ptr<gr::Runtime> held = self.start("Runtime.run_and_wait");
                try {
                    waitForRun(*held, std::nullopt);
                } catch (const py::error_already_set&) {
                    stopRun(*held);
                    throw;
                }
                raiseIfFailed(held->result());
            },
            "Runs the graph to its end on the runtime's thread and waits for it. Ctrl-C stops the run, waits for its end "
            "and raises KeyboardInterrupt, and a second Ctrl-C ends that wait and leaves the run ending; a run that "
            "fails raises its Error.")
        .def(
            "start", [](RuntimeObject& self) { std::ignore = self.start("Runtime.start"); },
            "Runs the graph on the runtime's thread, as run_in_thread() does, and returns. stop() ends the run; wait() "
            "and result() report it.")
        .def(
            "stop",
            [](RuntimeObject& self) {
                const std::shared_ptr<gr::Runtime> held = self.acquire();
                stopRun(*held);
            },
            "Stops the run in progress and waits for it to end. Without a run in progress it does nothing. Ctrl-C "
            "raises KeyboardInterrupt and leaves the stop requested and the run ending.")
        .def(
            "run_in_thread", [](RuntimeObject& self) { std::ignore = self.start("Runtime.run_in_thread"); }, "Runs the graph on the runtime's thread and returns; wait() and result() report the run.")
        .def(
            "wait",
            [](RuntimeObject& self, std::optional<double> timeout) {
                if (timeout.has_value() && std::isnan(*timeout)) {
                    throw py::value_error("wait() takes a timeout in seconds or None, got NaN");
                }
                std::optional<Clock::time_point> deadline;
                if (timeout.has_value() && *timeout < kLongestTimedWait) {
                    deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(*timeout, 0.0)));
                }
                std::shared_ptr<gr::Runtime> held;
                {
                    std::lock_guard lock(self.mutex);
                    held = self.runtime;
                }
                return held == nullptr || waitForRun(*held, deadline);
            },
            py::arg("timeout") = py::none(),
            "Waits for the last run to end. Returns True when it has ended, False when timeout seconds passed first. "
            " A NaN timeout raises ValueError; Ctrl-C raises KeyboardInterrupt and leaves the run "
            "going.")
        .def(
            "result", [](RuntimeObject& self) { raiseIfFailed(self.result()); }, "Returns None when the last run ended normally, and raises its Error otherwise. While a run is in progress it raises Error, and after a start whose thread did not start it raises that start's Error. After close() it reports the last run before the close.")
        .def_property_readonly(
            "state", [](RuntimeObject& self) { return self.acquire()->state(); }, "The scheduler's lifecycle state.")
        .def(
            "request_state",
            [](RuntimeObject& self, gr::Runtime::State state) {
                const std::shared_ptr<gr::Runtime> held = self.acquire();
                raiseIfFailed(withoutGil([&held, state] { return held->requestState(state); }));
            },
            py::arg("state"), "Requests a lifecycle transition of the scheduler.")
        .def(
            "poll_events", [](RuntimeObject& self, std::size_t maxEvents) { return self.acquire()->pollEvents(maxEvents); }, py::arg("max_events") = 64U, "Returns what the graph reported since the last call, oldest first, at most max_events of it.")
        .def(
            "send",
            [](RuntimeObject& self, gr::RuntimeCommand command, const std::string& service, const std::string& endpoint, const py::object& payload, const std::string& clientRequestId, const py::object& types) { //
                gr::property_map                   message = propertyMapFrom(payload, types);
                const std::shared_ptr<gr::Runtime> held    = self.acquire();
                raiseIfFailed(held->send(command, service, endpoint, std::move(message), clientRequestId));
            },
            py::arg("command"), py::arg("service"), py::arg("endpoint"), py::arg("payload") = py::none(), py::arg("client_request_id") = "", py::arg("types") = py::none(),
            "Sends one message to the scheduler's message input, addressed to service (a block's unique name) and "
            "endpoint; types maps a payload key to its type code. A message sent while the input is full is dropped and "
            "send() returns normally; a request dropped that way gets no reply, and a caller waiting for one times out.")
        .def(
            "quiesce", [](const py::object& owner) { return std::make_unique<QuiescenceObject>(owner, &owner.cast<RuntimeObject&>()); }, "Returns a context manager that quiesces the work loop for a live graph edit.")
        .def(
            "graph", [](RuntimeObject& self) { return holdWithoutGil(GraphObject{.graph = self.acquire()->graph(), .isView = true}); }, "Returns a view of the graph the scheduler holds.")
        .def(
            "scheduler", [](RuntimeObject& self) { return self.acquire()->scheduler(); }, "Returns the scheduler as a block, for its own settings.")
        .def(
            "close",
            [](RuntimeObject& self) {
                std::shared_ptr<gr::Runtime> held;
                bool                         quiesced = false;
                {
                    std::lock_guard lock(self.mutex);
                    quiesced = self.quiescing > 0UZ;
                    if (!quiesced) {
                        self.closing = true;
                        held         = self.runtime;
                    }
                }
                if (quiesced) {
                    raiseError("close() inside a quiesce() block", "Runtime.close");
                }
                if (held == nullptr) {
                    return;
                }
                stopRun(*held);
                withoutGil([&self, &held] {
                    {
                        std::lock_guard lock(self.mutex);
                        if (self.runtime != nullptr) {
                            self.closedResult = held->result();
                            self.runtime.reset();
                        }
                    }
                    held.reset();
                });
            },
            "Stops the run in progress, waits for its end and releases the runtime. A call on a closed runtime does "
            "nothing. Ctrl-C raises KeyboardInterrupt, leaves the stop requested and the run ending, and keeps the "
            "runtime, which a later close() releases.");
}

/// The key order of the graph writer; a merged document lists its keys in the order `to_yaml()` uses.
constexpr std::array<std::string_view, 13> kDocumentKeyOrder{"id", "version", "name", "unique_name", "block_category", "meta_information", "parameters", "ctx_parameters", "scheduler", "exported_ports", "blocks", "connections", "graph"};

/// One block entry of a graph document and the two names it carries.
struct DocumentBlock {
    Value::Map* entry      = nullptr;
    std::string name       = {};    // parameters.name, or the top-level name of a subgraph without one
    std::string uniqueName = {};    // unique_name
    bool        isSubgraph = false; // id: SUBGRAPH

    [[nodiscard]] std::string_view label() const { return name.empty() ? std::string_view(uniqueName) : std::string_view(name); }
};

/// Collects the block entries of a document's `blocks` list, and of the `blocks` list of every entry's `graph`, at
/// any depth.
void collectDocumentBlocks(Value::Map& graph, std::vector<DocumentBlock>& found) {
    const auto blocksIt = graph.find("blocks");
    auto*      blocks   = blocksIt == graph.end() ? nullptr : blocksIt->second.get_if<gr::Tensor<Value>>();
    if (blocks == nullptr) {
        return;
    }
    for (Value& block : *blocks) {
        auto* entry = block.get_if<Value::Map>();
        if (entry == nullptr) {
            continue;
        }
        DocumentBlock described{.entry = entry};
        if (const auto idIt = entry->find("id"); idIt != entry->end()) {
            described.isSubgraph = idIt->second.value_or(std::string()) == "SUBGRAPH";
        }
        bool namedInParameters = false;
        if (const auto parametersIt = entry->find("parameters"); parametersIt != entry->end()) {
            if (const auto* parameters = parametersIt->second.get_if<Value::Map>(); parameters != nullptr) {
                if (const auto nameIt = parameters->find("name"); nameIt != parameters->end()) {
                    described.name    = nameIt->second.value_or(std::string());
                    namedInParameters = true;
                }
            }
        }
        if (const auto nameIt = entry->find("name"); described.isSubgraph && !namedInParameters && nameIt != entry->end()) {
            described.name = nameIt->second.value_or(std::string());
        }
        if (const auto uniqueNameIt = entry->find("unique_name"); uniqueNameIt != entry->end()) {
            described.uniqueName = uniqueNameIt->second.value_or(std::string());
        }
        found.push_back(std::move(described));
        if (const auto graphIt = entry->find("graph"); graphIt != entry->end()) {
            if (auto* interior = graphIt->second.get_if<Value::Map>(); interior != nullptr) {
                collectDocumentBlocks(*interior, found);
            }
        }
    }
}

std::string joinLabels(const std::vector<const DocumentBlock*>& blocks) {
    std::string joined;
    for (const DocumentBlock* block : blocks) {
        joined += std::format("{}'{}'", joined.empty() ? "" : ", ", block->label());
    }
    return joined.empty() ? std::string("none") : joined;
}

/// Parses a graph document. Raises Error naming the line and column when it does not parse; `where` names the caller.
auto parseDocument(const std::string& document, const char* where) {
    auto parsed = withoutGil([&document] { return gr::pmt::yaml::deserialize(document); });
    if (!parsed.has_value()) {
        raiseError(std::format("line {}, column {}: {}", parsed.error().line, parsed.error().column, parsed.error().message), where);
    }
    return parsed;
}

/// Returns the one entry of `blocks` whose name or unique name is `blockName`. Raises Error when no entry or several
/// carry the name; `where` names the caller.
const DocumentBlock& namedEntry(const std::vector<DocumentBlock>& blocks, const std::string& blockName, const char* where) {
    std::vector<const DocumentBlock*> carriers;
    for (const DocumentBlock& block : blocks) {
        if (block.name == blockName || block.uniqueName == blockName) {
            carriers.push_back(&block);
        }
    }
    if (carriers.empty()) {
        std::vector<const DocumentBlock*> everyBlock;
        std::ranges::transform(blocks, std::back_inserter(everyBlock), [](const DocumentBlock& block) { return &block; });
        raiseError(std::format("no block of the document is named '{}'; its blocks are {}", blockName, joinLabels(everyBlock)), where);
    }
    if (carriers.size() > 1UZ) {
        raiseError(std::format("{} blocks of the document are named '{}': {}", carriers.size(), blockName, joinLabels(carriers)), where);
    }
    return *carriers.front();
}

/// Returns the map that holds the settings of `block`, where the loader reads them: the entry of a block, and the
/// `scheduler` of a subgraph. Raises Error for a subgraph without a scheduler, which takes no settings.
Value::Map& settingsHolder(const DocumentBlock& block, const std::string& blockName, const char* where) {
    if (!block.isSubgraph) {
        return *block.entry;
    }
    const auto  schedulerIt = block.entry->find("scheduler");
    Value::Map* holder      = schedulerIt == block.entry->end() ? nullptr : schedulerIt->second.get_if<Value::Map>();
    if (holder == nullptr) {
        raiseError(std::format("'{}' is a subgraph without a scheduler, which takes no settings", blockName), where);
    }
    return *holder;
}

/// Maps each of `names` to the type of the block entry of a graph document that carries the name, found as
/// `mergeParameters` finds it: the `id` of a block's entry, and the `id` of a subgraph's scheduler, which is the
/// registry key the loader instantiates. An entry without an `id` gives ''.
py::dict documentBlockTypes(const std::string& document, const py::iterable& names) {
    auto                       parsed = parseDocument(document, "document_block_types");
    std::vector<DocumentBlock> blocks;
    collectDocumentBlocks(*parsed, blocks);
    py::dict types;
    for (const py::handle item : names) {
        const std::string blockName = stringFrom(item, Subject{});
        const Value::Map& holder    = settingsHolder(namedEntry(blocks, blockName, "document_block_types"), blockName, "document_block_types");
        const auto        idIt      = holder.find("id");
        types[item]                 = idIt == holder.end() ? std::string() : idIt->second.value_or(std::string());
    }
    return types;
}

/// Sets `settings` (block name to key to value) in the block entries of a graph document and writes the document
/// again, with its type tags. A block is named by `parameters.name` or `unique_name`, and a subgraph without
/// `parameters.name` by its top-level `name`, as the graph loader names them. The settings of a block go to the
/// entry's `parameters`, and those of a subgraph with a scheduler to `scheduler.parameters`, where the loader reads
/// them; a subgraph without a scheduler takes no settings. A parameter held there takes the type of the value it
/// holds; one not held takes the type code `types` gives it (block name to key to code), or the type its Python value
/// selects.
std::string mergeParameters(const std::string& document, const py::dict& settings, const py::object& types) {
    if (!types.is_none() && !PyDict_Check(types.ptr())) {
        throw py::type_error(std::format("types need a dict of block names to dicts of type codes, got {}", typeNameOf(types)));
    }
    auto                       parsed = parseDocument(document, "merge_parameters");
    std::vector<DocumentBlock> blocks;
    collectDocumentBlocks(*parsed, blocks);

    for (const auto& [blockKey, parameters] : settings) {
        const std::string blockName = stringFrom(blockKey, Subject{});
        if (!PyDict_Check(parameters.ptr())) {
            throw py::type_error(std::format("the settings of '{}' need a dict, got {}", blockName, typeNameOf(parameters)));
        }
        Value::Map& holder          = settingsHolder(namedEntry(blocks, blockName, "merge_parameters"), blockName, "merge_parameters");
        Value&      parametersValue = holder["parameters"];
        if (!parametersValue.is_map()) {
            parametersValue = Value::Map{};
        }
        Value::Map& entryParameters = *parametersValue.get_if<Value::Map>();
        PyObject*   blockTypes      = types.is_none() ? nullptr : PyDict_GetItemWithError(types.ptr(), blockKey.ptr());
        throwIfPythonError();
        for (const auto& [key, item] : py::reinterpret_borrow<py::dict>(parameters)) {
            const std::string name = stringFrom(key, Subject{});
            const Subject     subject{.key = name};
            const auto        held     = entryParameters.find(name);
            PyObject*         typeCode = blockTypes == nullptr || !PyDict_Check(blockTypes) ? nullptr : PyDict_GetItemWithError(blockTypes, key.ptr());
            throwIfPythonError();
            Value value;
            if (held != entryParameters.end() && !held->second.is_monostate()) {
                value = valueAs(item, parseTypeCode(typeCodeOf(held->second)), subject);
            } else if (typeCode != nullptr) {
                value = valueAs(item, parseTypeCode(stringFrom(typeCode, subject)), subject);
            } else {
                value = valueFrom(item, subject);
            }
            entryParameters.insert_or_assign(std::pmr::string(name), std::move(value));
        }
    }
    return withoutGil([&parsed] { return gr::pmt::yaml::serialize<gr::pmt::yaml::TypeTagMode::Auto>(*parsed, kDocumentKeyOrder); });
}

} // namespace

PYBIND11_MODULE(_gr4, module) {
    module.doc() = "The type-erased runtime of gnuradio4-core: graphs built by registry name, block settings, schedulers "
                   "and the events a running graph reports.";

    applyDefaultPluginDirectory();

    module.def(
        "convert", [](const py::object& object, const std::optional<std::string>& typeCode) { return toPython(convertFrom(object, typeCode)); }, py::arg("obj"), py::arg("type_code") = py::none(),
        "Converts obj to a pmt value and back to Python.\n\n"
        "Without type_code the Python type selects the value type. With type_code the value has exactly that type: a "
        "value out of range, or a fraction for an integer type, raises OverflowError, and one of the wrong kind "
        "TypeError.");
    module.def("type_code", [](const py::object& object, const std::optional<std::string>& typeCode) { return typeCodeOf(convertFrom(object, typeCode)); }, py::arg("obj"), py::arg("type_code") = py::none(), "Returns the type code of the pmt value obj converts to, by the rule convert() applies.");

    defineErrorClass(module);
    py::register_local_exception_translator(&translateException);
    defineEnumerations(module);
    defineBlockHandle(module);
    definePortAndEdge(module);
    defineRuntimeGraph(module);
    defineRuntimeEvent(module);
    defineRuntime(module);

    module.attr("DEFAULT_SCHEDULER") = std::string(gr::Runtime::kDefaultScheduler);
    module.def(
        "available_block_types",
        [] {
            noteLoaderUse();
            return withoutGil([] { return gr::RuntimeGraph::availableBlockTypes(); });
        },
        "Returns the registry key of every block type this process can build.");
    module.def(
        "available_scheduler_types",
        [] {
            noteLoaderUse();
            return withoutGil([] { return gr::RuntimeGraph::availableSchedulerTypes(); });
        },
        "Returns the registry key of every scheduler type this process can build.");
    module.def("merge_parameters", &mergeParameters, py::arg("document"), py::arg("settings"), py::arg("types") = py::none(),
        "Sets settings (block name to key to value) in the block entries of a graph document and returns the document "
        "written again with its type tags.\n\n"
        "A block entry at any depth of subgraphs is named by its parameters.name or its unique_name, and a subgraph "
        "without parameters.name by its top-level name. The settings of a subgraph with a scheduler go to its "
        "scheduler.parameters; a subgraph without a scheduler takes no settings and raises Error. A parameter held "
        "there takes the type of the value it holds; one not held takes the type code types gives it (block name to "
        "key to code), or the type its Python value selects. A name no entry carries, or one two entries carry, raises "
        "Error; so does a document that does not parse, naming the line and column.");
    module.def("document_block_types", &documentBlockTypes, py::arg("document"), py::arg("names"),
        "Maps each of names to the type of the block entry of a graph document that carries the name.\n\n"
        "Entries are named as merge_parameters() names them. The type of a block is the id of its entry, and that of a "
        "subgraph with a scheduler the id of the scheduler; an entry without an id gives ''. A name no entry carries, "
        "one two entries carry, or one of a subgraph without a scheduler raises Error; so does a document that does not "
        "parse, naming the line and column.");
    module.def("plugin_directories", &pluginDirectories, "Returns the directories the plugin loader reads, from GNURADIO4_PLUGIN_DIRECTORIES as it was at the loader's first use.");
}
