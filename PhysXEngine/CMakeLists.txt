find_package(GTest REQUIRED)
find_package(OpenGL REQUIRED)

add_library(metis_core
  src/SimState.cpp
  src/GravityCpu.cpp
  src/SemiImplicitEuler.cpp
  src/LeapfrogKdk.cpp
  src/Scenario.cpp
  src/Diagnostics.cpp
)
target_include_directories(metis_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(metis_cuda cuda/GravityGpu.cu)
target_include_directories(metis_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
target_link_libraries(metis_cuda PUBLIC metis_core)
target_link_libraries(metis_cuda PRIVATE OpenGL::GL)
set_target_properties(metis_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_executable(metis_tests
  tests/test_kepler.cpp
  tests/test_leapfrog.cpp
  tests/test_diagnostics.cpp
  tests/test_gpu_vs_cpu.cpp
)
target_link_libraries(metis_tests PRIVATE metis_core metis_cuda GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(metis_tests)
