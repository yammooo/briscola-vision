cmake_minimum_required(VERSION 3.16)
project(briscola_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(briscola
    src/evaluation.cpp
    src/game.cpp
    src/io.cpp
    src/rules.cpp
)
target_include_directories(briscola PUBLIC include)
