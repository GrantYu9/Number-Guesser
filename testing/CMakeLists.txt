set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/testing)

enable_testing()

find_package(GTest)

add_executable(test_activation_functions test_activation_functions.cpp)
target_link_libraries(test_activation_functions GTest::gtest_main internal)

add_executable(test_backpropagation test_backpropagation.cpp)
target_link_libraries(test_backpropagation GTest::gtest_main internal)

add_executable(test_dense_layer test_dense_layer.cpp)
target_link_libraries(test_dense_layer GTest::gtest_main internal)

add_executable(test_input test_input.cpp)
target_link_libraries(test_input GTest::gtest_main internal)

add_executable(test_loss_functions test_loss_functions.cpp)
target_link_libraries(test_loss_functions GTest::gtest_main internal)

add_executable(test_neural_network test_neural_network.cpp)
target_link_libraries(test_neural_network GTest::gtest_main internal)

add_executable(test_persistence test_persistence.cpp)
target_link_libraries(test_persistence GTest::gtest_main internal)

add_executable(test_stochastic_gradient_descent test_stochastic_gradient_descent.cpp)
target_link_libraries(test_stochastic_gradient_descent GTest::gtest_main internal)

include(GoogleTest)

gtest_discover_tests(
    test_activation_functions
    test_backpropagation
    test_dense_layer
    test_input
    test_loss_functions
    test_neural_network
    test_persistence
    test_stochastic_gradient_descent 
)
