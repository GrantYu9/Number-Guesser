add_library(core INTERFACE)
add_library(core_compiled
    ${CMAKE_SOURCE_DIR}/src/activation_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/dense_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/exceptions.cpp
    ${CMAKE_SOURCE_DIR}/src/input.cpp
    ${CMAKE_SOURCE_DIR}/src/loss_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/neural_network.cpp
    ${CMAKE_SOURCE_DIR}/src/persistence.cpp
)
add_library(core_header_only INTERFACE)

target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core_compiled PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(core_compiled PRIVATE external)
target_link_libraries(core INTERFACE core_compiled core_header_only)
