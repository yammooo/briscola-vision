cmake_minimum_required(VERSION 3.16)
project(briscola_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgcodecs imgproc videoio)

add_library(briscola
    src/app.cpp
    src/debug.cpp
    src/evaluation.cpp
    src/game.cpp
    src/io.cpp
    src/briscola_providers/most_frequent_briscola_provider.cpp
    src/rules.cpp
    src/round_analyzers/yolo_sift_round_analyzer.cpp
)
target_include_directories(briscola PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(briscola PUBLIC ${OpenCV_LIBS})

add_executable(yolo_sift apps/yolo_sift.cpp)
target_link_libraries(yolo_sift PRIVATE briscola)

include(CTest)
if(BUILD_TESTING)
    add_executable(briscola_tests tests/most_frequent_briscola_provider_test.cpp)
    target_link_libraries(briscola_tests PRIVATE briscola)
    add_test(NAME briscola_tests COMMAND briscola_tests)

    add_executable(yolo_detector_smoke tests/yolo_card_detector_smoke.cpp)
    target_link_libraries(yolo_detector_smoke PRIVATE briscola)
    if(EXISTS "${CMAKE_SOURCE_DIR}/models/briscola_cards.onnx")
        add_test(
            NAME yolo_detector_smoke
            COMMAND yolo_detector_smoke
                "${CMAKE_SOURCE_DIR}/models/briscola_cards.onnx"
                "${CMAKE_SOURCE_DIR}/data/Briscola_Trentine/1-clubs.JPG"
        )
    endif()
endif()
