cmake_minimum_required(VERSION 3.12)
project(image_distanc)

set(CMAKE_CXX_STANDARD 14)

add_executable(image_distanc main.cpp)
target_link_libraries(image_distanc
        "-Wl,-Bdynamic"
        -lopencv_core
        -lopencv_imgproc
        -lopencv_imgcodecs
        )