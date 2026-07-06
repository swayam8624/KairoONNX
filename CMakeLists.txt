cmake_minimum_required(VERSION 3.28)
project(KairoONNX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(KairoONNX)
target_sources(KairoONNX
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES src/KairoONNX.cppm
)

add_executable(KairoONNXSmoke tests/smoke.cpp)
target_link_libraries(KairoONNXSmoke PRIVATE KairoONNX)

enable_testing()
add_test(NAME KairoONNXSmoke COMMAND KairoONNXSmoke)
