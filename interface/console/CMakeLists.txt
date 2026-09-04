set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/console)
set(CMAKE_BUILD_TYPE Debug)

add_executable(console console.cpp)

target_link_libraries(console include)
