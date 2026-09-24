/**
 * @brief The extension module gr4._gr4: the type-erased runtime of gnuradio4-core for Python.
 *
 * The module binds `gnuradio-4.0/Runtime.hpp` one class to one class. Blocks and schedulers are named by their
 * registry keys, and settings travel as `pmt::Value` maps converted to and from Python objects.
 */

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Runtime.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/Value.hpp>

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
#include <limits>
#include <memory>
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
/// plugin loader reads the variable once, at the first registry call, and searches a path relative to the working
/// directory without it. Setting it through `os.environ` gives Python and the loader one value.
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
            "active_parameters", [](const gr::BlockHandle& block) { return mapToPython(block.activeParameters()); }, "Returns the settings in force.")
        .def(
            "staged_parameters", [](const gr::BlockHandle& block) { return mapToPython(block.stagedParameters()); }, "Returns the settings staged for the block's next work call.")
        .def(
            "default_parameters", [](const gr::BlockHandle& block) { return mapToPython(block.defaultParameters()); }, "Returns the settings reset_defaults() restores.")
        .def(
            "setting_types",
            [](const gr::BlockHandle& block) {
                py::dict types;
                for (const auto& [key, value] : block.activeParameters()) {
                    types[stringToPython(key)] = typeCodeOf(value);
                }
                return types;
            },
            "Returns the type code of every active setting, by key.")
        .def(
            "set_staged", [](gr::BlockHandle& block, const py::object& parameters, const py::object& types) { return mapToPython(block.setStaged(propertyMapFrom(parameters, types))); }, py::arg("parameters"), py::arg("types") = py::none(), "Stages settings for the block's next work call; types maps a key to its type code. Returns the settings that could not be staged.")
        .def(
            "set", [](gr::BlockHandle& block, const py::object& parameters, const py::object& types) { return mapToPython(block.set(propertyMapFrom(parameters, types))); }, py::arg("parameters"), py::arg("types") = py::none(), "Stores settings without staging them; types maps a key to its type code. Returns the settings that could not be stored.")
        .def(
            "meta_information", [](const gr::BlockHandle& block) { return mapToPython(block.metaInformation()); }, "Returns the block's meta-information.")
        .def(
            "set_meta_information", [](gr::BlockHandle& block, const py::object& information) { block.setMetaInformation(propertyMapFrom(information, py::none())); }, py::arg("information"), "Replaces the block's meta-information.")
        .def("store_defaults", &gr::BlockHandle::storeDefaults, "Makes the active settings the defaults.")
        .def("reset_defaults", &gr::BlockHandle::resetDefaults, "Restores the default settings.")
        .def(
            "__eq__", [](const gr::BlockHandle& self, const gr::BlockHandle& other) { return self.uniqueName() == other.uniqueName(); }, py::is_operator())
        .def("__hash__", [](const gr::BlockHandle& block) { return py::hash(py::str(std::string(block.uniqueName()))); })
        .def("__repr__", &blockRepr);
}

void definePortAndEdge(py::module_& module) {
    py::class_<gr::RuntimePort>(module, "RuntimePort", "One port of a block, as RuntimeGraph.input_ports() and output_ports() describe it.").def_readonly("name", &gr::RuntimePort::name, "The name connect() accepts: 'out', or 'in#1' for element 1 of a collection.").def_readonly("type_name", &gr::RuntimePort::typeName, "The value type, as port_type_name() spells it.").def_readonly("is_input", &gr::RuntimePort::isInput, "True for an input port.").def_readonly("is_message", &gr::RuntimePort::isMessage, "True for a message port, False for a stream port.").def_readonly("is_optional", &gr::RuntimePort::isOptional, "True when the port may stay unconnected.").def_readonly("is_synchronous", &gr::RuntimePort::isSynchronous, "True when the port moves samples in step with the block's other synchronous ports.").def_readonly("is_connected", &gr::RuntimePort::isConnected, "True while a running scheduler holds the port's edge connected.").def_readonly("domain", &gr::RuntimePort::domain, "The compute domain of the port.").def_readonly("collection", &gr::RuntimePort::collection, "The name of the collection the port belongs to, or ''.").def_readonly("index", &gr::RuntimePort::index, "The port's position in its collection, or 0.").def_readonly("min_samples", &gr::RuntimePort::minSamples, "The fewest samples the port takes per work call.").def_readonly("max_samples", &gr::RuntimePort::maxSamples, "The most samples the port takes per work call.").def("__repr__", [](const gr::RuntimePort& port) { return std::format("<RuntimePort {} {} {}>", port.name, port.isInput ? "input" : "output", port.typeName); });

    py::class_<gr::RuntimeEdge, GilFreeHolder<gr::RuntimeEdge>>(module, "RuntimeEdge", "One edge of a graph, as RuntimeGraph.edges() lists it.").def_property_readonly("source_block", [](const gr::RuntimeEdge& edge) { return edge.sourceBlock; }, "The block the edge leaves.").def_readonly("source_port", &gr::RuntimeEdge::sourcePort, "The output port the edge leaves.").def_property_readonly("destination_block", [](const gr::RuntimeEdge& edge) { return edge.destinationBlock; }, "The block the edge enters.").def_readonly("destination_port", &gr::RuntimeEdge::destinationPort, "The input port the edge enters.").def_property_readonly("min_buffer_size", [](const gr::RuntimeEdge& edge) { return edge.edge.minBufferSize; }, "The smallest buffer the edge asks for, the framework's default when none was given.").def_property_readonly("weight", [](const gr::RuntimeEdge& edge) { return edge.edge.weight; }, "The edge's weight.").def_property_readonly("name", [](const gr::RuntimeEdge& edge) { return edge.edge.name; }, "The edge's name.").def("__repr__", [](const gr::RuntimeEdge& edge) { return std::format("<RuntimeEdge {}.{} -> {}.{}>", edge.sourceBlock.uniqueName(), edge.sourcePort, edge.destinationBlock.uniqueName(), edge.destinationPort); });
}

void defineRuntimeGraph(py::module_& module) {
    py::class_<GraphObject, GilFreeHolder<GraphObject>>(module, "RuntimeGraph",
        "A flow graph built by registry name.\n\n"
        "A RuntimeGraph owns its graph, or views one a subgraph block or a scheduler owns. A graph handed to "
        "Runtime.create moves into the scheduler, and every later call on this object raises Error.\n\n"
        "The constructor, emplace(), emplace_subgraph(), remove(), clear() and from_yaml() release the GIL, and so "
        "does the deallocation of a graph, a block handle or an edge.")
        .def(py::init([](const py::object& settings, const py::object& types) {
            gr::property_map initialSettings = propertyMapFrom(settings, types);
            noteLoaderUse();
            return holdWithoutGil(GraphObject{.graph = withoutGil([&initialSettings] { return gr::RuntimeGraph(std::move(initialSettings)); })});
        }),
            py::arg("settings") = py::none(), py::arg("types") = py::none(), "Creates an empty graph with the given graph settings.")
        .def(
            "emplace",
            [](GraphObject& self, const std::string& type, const std::string& name, const py::object& parameters, const py::object& types) {
                gr::property_map  initialParameters = propertyMapFrom(parameters, types);
                gr::RuntimeGraph& graph             = self.get();
                return valueOrRaise(withoutGil([&] { return graph.emplace(type, name, std::move(initialParameters)); }));
            },
            py::arg("type"), py::arg("name"), py::arg("parameters") = py::none(), py::arg("types") = py::none(), "Instantiates the block type registered as type, names it and adds it. The parameters are in force when the block starts; types maps a key to its type code.")
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
            "clear",
            [](GraphObject& self) {
                gr::RuntimeGraph& graph = self.get();
                withoutGil([&graph] { graph.clear(); });
            },
            "Removes every block and edge.")
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
            "port_type_name", [](GraphObject& self, const gr::BlockHandle& block, bool isInput, const std::string& port) { return self.get().portTypeName(block, isInput, port); }, py::arg("block"), py::arg("is_input"), py::arg("port"), "Returns the value type of one port, or '' when the block has no such port.")
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
    defineBlockHandle(module);
    definePortAndEdge(module);
    defineRuntimeGraph(module);

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
    module.def("plugin_directories", &pluginDirectories, "Returns the directories the plugin loader reads, from GNURADIO4_PLUGIN_DIRECTORIES as it was at the loader's first use.");
}
