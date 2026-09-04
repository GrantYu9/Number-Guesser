add_subdirectory(core)

add_library(internal INTERFACE)

target_link_libraries(internal INTERFACE core)
