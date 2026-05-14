cmake_minimum_required(VERSION 3.20)

project(matrix_acc VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src)
<<<<<<< HEAD
enable_testing()
add_subdirectory(test)
=======
add_subdirectory(test)

enable_testing()
>>>>>>> a6769ae18341c128571217d4b26a37721c1ccc3f
