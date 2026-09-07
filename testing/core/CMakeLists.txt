set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/testing/core)

add_executable(test_dense_layer test_dense_layer.cpp)
target_link_libraries(test_dense_layer GTest::gtest_main include)

include(GoogleTest)

gtest_discover_tests(test_dense_layer)
