add_library(core INTERFACE)
add_library(core_compiled
    ${CMAKE_SOURCE_DIR}/src/core/activation_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/core/dense_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/core/input.cpp
    ${CMAKE_SOURCE_DIR}/src/core/neural_network.cpp
)
add_library(core_header_only INTERFACE)

target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core_compiled PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core_header_only INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(core_compiled PRIVATE external)
target_link_libraries(core INTERFACE core_compiled core_header_only)
