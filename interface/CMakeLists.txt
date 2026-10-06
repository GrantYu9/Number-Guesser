set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/interface)

# set(CMAKE_BUILD_TYPE Debug)

find_package(spdlog REQUIRED)

add_executable(console console.cpp)
target_link_libraries(console PRIVATE internal)

add_executable(initialization initialization.cpp)
target_link_libraries(initialization PRIVATE internal)

add_executable(testing testing.cpp)
target_link_libraries(testing PRIVATE internal)
target_link_libraries(testing PRIVATE utils)
target_link_libraries(testing PRIVATE spdlog::spdlog)

add_executable(training training.cpp)
target_link_libraries(training PRIVATE internal)
target_link_libraries(training PRIVATE utils)
target_link_libraries(training PRIVATE spdlog::spdlog)
