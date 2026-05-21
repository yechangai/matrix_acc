# CMake generated Testfile for 
# Source directory: /home/chang/Desktop/matrix_acc/test
# Build directory: /home/chang/Desktop/matrix_acc/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[allocator_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_allocator")
set_tests_properties([=[allocator_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;18;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[mat_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_mat")
set_tests_properties([=[mat_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;36;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[cuda_allocator_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_cuda_allocator")
set_tests_properties([=[cuda_allocator_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;67;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[cuda_mat_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_cuda_mat")
set_tests_properties([=[cuda_mat_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;86;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[matmul_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_matmul")
set_tests_properties([=[matmul_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;113;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[gemm_perf_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_gemm_perf")
set_tests_properties([=[gemm_perf_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;157;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
add_test([=[compare_test]=] "/home/chang/Desktop/matrix_acc/build/test/test_compare")
set_tests_properties([=[compare_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;198;add_test;/home/chang/Desktop/matrix_acc/test/CMakeLists.txt;0;")
