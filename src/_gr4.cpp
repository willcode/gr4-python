/**
 * @brief The extension module gr4._gr4: the type-erased runtime of gnuradio4-core for Python.
 *
 * The module binds `gnuradio-4.0/Runtime.hpp` one class to one class. Blocks and schedulers are named by their
 * registry keys, and settings travel as `pmt::Value` maps converted to and from Python objects.
 */

#include <pybind11/pybind11.h>

#include <cstdlib>

namespace py = pybind11;

namespace {

constexpr const char* kPluginDirectoriesVariable = "GNURADIO4_PLUGIN_DIRECTORIES";

/// Names the plugin directory of the core this module was built against when the environment names none. The
/// plugin loader reads the variable once, at the first registry call, and searches a path relative to the working
/// directory without it. Setting it through `os.environ` gives Python and the loader one value.
void applyDefaultPluginDirectory() {
    if (std::getenv(kPluginDirectoriesVariable) == nullptr) {
        py::module_::import("os").attr("environ")[kPluginDirectoriesVariable] = GR4_DEFAULT_PLUGIN_DIRECTORY;
    }
}

} // namespace

PYBIND11_MODULE(_gr4, module) {
    module.doc() = "The type-erased runtime of gnuradio4-core: graphs built by registry name, block settings, schedulers "
                   "and the events a running graph reports.";

    applyDefaultPluginDirectory();
}
