CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

project(linear_regression)

find_package(Torch REQUIRED)
find_package(Matplot++ REQUIRED)


add_executable(linear_regression main.cpp)
target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
target_link_libraries(linear_regression Matplot++::matplot)


set_property(TARGET linear_regression PROPERTY CXX_STANDARD 20)

INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})

SET(directory IMAGE_data)

add_custom_target(build-time-make-directory ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${directory})