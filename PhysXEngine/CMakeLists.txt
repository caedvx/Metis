find_package(GTest REQUIRED)

add_library(metis_core
  src/SimState.cpp
  src/GravityCpu.cpp
  src/SemiImplicitEuler.cpp
  src/LeapfrogKdk.cpp
  src/Scenario.cpp
  src/Diagnostics.cpp
)
target_include_directories(metis_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(metis_tests
  tests/test_kepler.cpp
  tests/test_leapfrog.cpp
  tests/test_diagnostics.cpp
)
target_link_libraries(metis_tests PRIVATE metis_core GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(metis_tests)
