# CMake generated Testfile for 
# Source directory: /home/chang/Desktop/matrix_acc/test
# Build directory: /home/chang/Desktop/matrix_acc/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[allocator_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_allocator")
set_tests_properties([=[allocator_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;18;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[cuda_allocator_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_cuda_allocator")
set_tests_properties([=[cuda_allocator_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;41;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[compare_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_compare")
set_tests_properties([=[compare_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;64;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
