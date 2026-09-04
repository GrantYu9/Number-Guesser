add_subdirectory(external)
add_subdirectory(internal)

add_library(include INTERFACE)

target_link_libraries(include INTERFACE external internal)
