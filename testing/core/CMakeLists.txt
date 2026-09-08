set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/testing/core)

# set(CMAKE_BUILD_TYPE Debug)

add_executable(test_activation_functions test_activation_functions.cpp)
target_link_libraries(test_activation_functions GTest::gtest_main core)

add_executable(test_dense_layer test_dense_layer.cpp)
target_link_libraries(test_dense_layer GTest::gtest_main core)

add_executable(test_input test_input.cpp)
target_link_libraries(test_input GTest::gtest_main core)

add_executable(test_persistence test_persistence.cpp)
target_link_libraries(test_persistence GTest::gtest_main core)

include(GoogleTest)

gtest_discover_tests(
    test_activation_functions
    test_dense_layer
    test_input
    test_persistence
)
