"""Tests of the registry and of the plugin directories it loads from."""

import os
import subprocess
import sys
import unittest

from gr4 import _gr4


def mapped_files():
    """Returns the files this process maps."""
    with open("/proc/self/maps", encoding="utf-8") as maps:
        return {fields[5] for fields in (line.split() for line in maps) if len(fields) >= 6}


def plugin_files(paths):
    """Returns the paths inside the directories GNURADIO4_PLUGIN_DIRECTORIES names."""
    named = [os.path.realpath(entry) for entry in os.environ["GNURADIO4_PLUGIN_DIRECTORIES"].split(":") if entry]
    return {path for path in paths if any(os.path.realpath(path).startswith(entry + os.sep) for entry in named)}


# Read at import, before any call reaches the plugin loader.
MAPPED_AT_IMPORT = mapped_files()

PLUGIN_BLOCKS = [
    "gr::blocks::testing::CountingSource<float32>",
    "gr::blocks::testing::HeadBlock<float32>",
    "gr::blocks::testing::NullSource<float32>",
    "gr::blocks::testing::NullSink<float32>",
    "gr::blocks::fileio::BasicFileSink<float32>",
]


class Registry(unittest.TestCase):
    """The registry lists what this process can build."""

    def test_block_types_include_the_plugin_blocks(self):
        block_types = _gr4.available_block_types()
        for key in PLUGIN_BLOCKS:
            self.assertIn(key, block_types)

    def test_scheduler_types_include_the_default(self):
        self.assertIn(_gr4.DEFAULT_SCHEDULER, _gr4.available_scheduler_types())


class PluginDirectories(unittest.TestCase):
    """The loader reads the directories GNURADIO4_PLUGIN_DIRECTORIES names, or the build's default."""

    def test_plugin_directories_follow_the_environment_as_the_loader_first_read_it(self):
        _gr4.available_block_types()
        named = [directory for directory in os.environ["GNURADIO4_PLUGIN_DIRECTORIES"].split(":") if directory]
        self.assertTrue(named)
        self.assertEqual(_gr4.plugin_directories(), named)
        self.addCleanup(
            os.environ.__setitem__, "GNURADIO4_PLUGIN_DIRECTORIES", os.environ["GNURADIO4_PLUGIN_DIRECTORIES"]
        )
        os.environ["GNURADIO4_PLUGIN_DIRECTORIES"] = os.path.join(named[0], "set-after-the-first-registry-call")
        self.assertEqual(_gr4.plugin_directories(), named)

    def test_import_names_the_default_directory_when_the_environment_names_none(self):
        environment = dict(os.environ)
        del environment["GNURADIO4_PLUGIN_DIRECTORIES"]
        program = (
            "import os, sys\n"
            "from gr4 import _gr4\n"
            "print(os.environ['GNURADIO4_PLUGIN_DIRECTORIES'])\n"
            "print(':'.join(_gr4.plugin_directories()))\n"
            "sys.stdout.flush()\n"
            "os._exit(0)\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", program], env=environment, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        variable, directories = run.stdout.splitlines()
        self.assertTrue(variable.endswith("gnuradio-4/plugins"), variable)
        self.assertEqual(directories, variable)


class LoaderAlone(unittest.TestCase):
    """The plugin loader alone opens the plugins: nothing preloads one, and the extension opens without RTLD_GLOBAL."""

    def test_no_plugin_is_mapped_before_the_loader_opens_them_and_one_registry_library_after(self):
        self.assertEqual(plugin_files(MAPPED_AT_IMPORT), set())
        self.assertEqual(sys.getdlopenflags() & os.RTLD_GLOBAL, 0)
        self.assertNotIn("gnuradio", os.environ.get("LD_PRELOAD", ""))
        _gr4.available_block_types()
        mapped = mapped_files()
        self.assertTrue(plugin_files(mapped))
        registry = {os.path.realpath(path) for path in mapped if "libgnuradio-blocklib-core" in os.path.basename(path)}
        self.assertEqual(len(registry), 1, registry)


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
