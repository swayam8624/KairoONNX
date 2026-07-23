cmake_minimum_required(VERSION 3.28)
project(KairoONNX LANGUAGES CXX)

set(KAIRO_FOUNDATION_DIR "/Users/swayamsingal/Desktop/Programming/Kairo/Foundation" CACHE PATH "Path to Kairo Foundation.")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(KairoONNX)
add_subdirectory("${KAIRO_FOUNDATION_DIR}/KairoMath" "${CMAKE_BINARY_DIR}/KairoMath")
target_sources(KairoONNX
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES src/KairoONNX.cppm
)
target_link_libraries(KairoONNX PUBLIC KairoMath)

add_executable(KairoONNXSmoke tests/smoke.cpp)
target_link_libraries(KairoONNXSmoke PRIVATE KairoONNX)

enable_testing()
add_test(NAME KairoONNXSmoke COMMAND KairoONNXSmoke)
