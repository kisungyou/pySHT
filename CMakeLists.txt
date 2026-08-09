cmake_minimum_required(VERSION 3.26...4.0)

project(pysht LANGUAGES CXX)

find_package(
  Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _core
  STABLE_ABI
  NB_STATIC
  cpp/core.cpp
)

target_compile_features(_core PRIVATE cxx_std_17)
target_compile_definitions(
  _core
  PRIVATE PYSHT_VERSION="${SKBUILD_PROJECT_VERSION_FULL}"
)

if(MSVC)
  target_compile_options(_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(
  TARGETS _core
  LIBRARY DESTINATION pysht
  RUNTIME DESTINATION pysht
  ARCHIVE DESTINATION pysht
)
