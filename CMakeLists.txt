cmake_minimum_required(VERSION 3.25)
project(gr4-python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

find_package(gnuradio4 CONFIG REQUIRED)
find_package(
  Python 3.12
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_package(pybind11 3 CONFIG REQUIRED)

# The core's package configuration sits in <libdir>/cmake/gnuradio4 of its
# prefix, and the core installs block plugins in <libdir>/gnuradio-4/plugins.
cmake_path(GET gnuradio4_DIR PARENT_PATH GR4_CMAKE_DIR)
cmake_path(GET GR4_CMAKE_DIR PARENT_PATH GR4_LIBRARY_DIR)
set(GR4_DEFAULT_PLUGIN_DIRECTORY "${GR4_LIBRARY_DIR}/gnuradio-4/plugins")

set(GR4_TEST_PLUGIN_DIRECTORIES
    "${GR4_DEFAULT_PLUGIN_DIRECTORY}"
    CACHE STRING
          "colon-separated plugin directories the tests load blocks from")

set(GR4_PACKAGE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/gr4")
set(GR4_PACKAGE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/gr4"
)

pybind11_add_module(_gr4 src/_gr4.cpp)
target_link_libraries(_gr4 PRIVATE gnuradio4::gnuradio-core
                                   gnuradio4::gnuradio-blocklib-core)
target_compile_definitions(
  _gr4 PRIVATE GR4_DEFAULT_PLUGIN_DIRECTORY="${GR4_DEFAULT_PLUGIN_DIRECTORY}")
target_compile_options(_gr4 PRIVATE -Wall -Wextra)
set_target_properties(
  _gr4 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${GR4_PACKAGE_BUILD_DIR}"
                  INSTALL_RPATH "${GR4_LIBRARY_DIR}")

# The build tree holds an importable package: the extension beside copies of the
# package's Python files.
file(
  GLOB GR4_PACKAGE_FILES
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/gr4/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/python/gr4/*.pyi"
  "${CMAKE_CURRENT_SOURCE_DIR}/python/gr4/py.typed")
foreach(package_file IN LISTS GR4_PACKAGE_FILES)
  cmake_path(GET package_file FILENAME package_file_name)
  configure_file("${package_file}"
                 "${GR4_PACKAGE_BUILD_DIR}/${package_file_name}" COPYONLY)
endforeach()

install(TARGETS _gr4 LIBRARY DESTINATION "${GR4_PACKAGE_INSTALL_DIR}")
install(FILES ${GR4_PACKAGE_FILES} DESTINATION "${GR4_PACKAGE_INSTALL_DIR}")

enable_testing()
file(GLOB GR4_TEST_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py")
foreach(test_file IN LISTS GR4_TEST_FILES)
  cmake_path(GET test_file STEM test_name)
  add_test(
    NAME ${test_name}
    COMMAND Python::Interpreter "${test_file}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  set_tests_properties(
    ${test_name}
    PROPERTIES
      ENVIRONMENT
      "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python;GNURADIO4_PLUGIN_DIRECTORIES=${GR4_TEST_PLUGIN_DIRECTORIES};PYTHONDONTWRITEBYTECODE=1"
  )
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endforeach()
