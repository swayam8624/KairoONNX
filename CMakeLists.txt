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
        FILES
            src/KairoONNX.cppm
            src/KairoONNXRuntime.cppm
            src/KairoONNXOptimize.cppm
)
target_link_libraries(KairoONNX PUBLIC KairoMath)

add_executable(KairoONNXSmoke tests/smoke.cpp)
target_link_libraries(KairoONNXSmoke PRIVATE KairoONNX)
target_compile_definitions(
    KairoONNXSmoke PRIVATE KAIRO_ONNX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

enable_testing()
add_test(NAME KairoONNXSmoke COMMAND KairoONNXSmoke)

# Kairo portfolio warning policy: Kairo-owned source warnings are build failures.
# Catch2's __COUNTER__ registration macro is the only bounded Clang 23 test
# exception; production targets receive no blanket warning suppression.
foreach(_kairo_target IN ITEMS KairoONNX KairoONNXSmoke)
    if(TARGET ${_kairo_target})
        set_property(TARGET ${_kairo_target} PROPERTY COMPILE_WARNING_AS_ERROR ON)
    endif()
endforeach()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 23)
    foreach(_kairo_test_target IN ITEMS KairoONNXSmoke)
        if(TARGET ${_kairo_test_target})
            target_compile_options(${_kairo_test_target} PRIVATE -Wno-c2y-extensions)
        endif()
    endforeach()
endif()
