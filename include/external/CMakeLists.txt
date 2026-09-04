find_package(Eigen3 REQUIRED)

add_library(external INTERFACE)

target_include_directories(external INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(external INTERFACE Eigen3::Eigen)
