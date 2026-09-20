find_package(GTest REQUIRED)

add_library(metis_core
  ${CMAKE_CURRENT_SOURCE_DIR}/Particle.cpp
)
target_include_directories(metis_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(metis_tests
  tests/test_kepler.cpp
)
target_link_libraries(metis_tests PRIVATE metis_core GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(metis_tests)
