add_library(internal INTERFACE)
add_library(internal_compiled
    ${CMAKE_SOURCE_DIR}/src/activation_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/backpropagation.cpp
    ${CMAKE_SOURCE_DIR}/src/dense_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/exceptions.cpp
    ${CMAKE_SOURCE_DIR}/src/input.cpp
    ${CMAKE_SOURCE_DIR}/src/loss_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/neural_network.cpp
    ${CMAKE_SOURCE_DIR}/src/persistence.cpp
    ${CMAKE_SOURCE_DIR}/src/stochastic_gradient_descent.cpp
)
add_library(internal_header_only INTERFACE)

target_include_directories(internal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(internal_compiled PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(internal_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(internal_compiled PRIVATE external)
target_link_libraries(internal INTERFACE internal_compiled internal_header_only)
