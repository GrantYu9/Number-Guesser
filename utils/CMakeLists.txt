add_library(utils ${CMAKE_SOURCE_DIR}/utils/logger.cpp)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utils PRIVATE internal)
