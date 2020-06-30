cmake_minimum_required(VERSION 3.9.2)
project(eddl_benchmarks)
find_package(EDDL REQUIRED)

# Place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

#Here we can add the examples
#set(EDDL_PATH eddl/)

add_executable(eddl_mnist_mlp eddl/eddl_mnist_mlp.cpp)
target_link_libraries(eddl_mnist_mlp PUBLIC EDDL::eddl)

add_executable(eddl_mnist_conv eddl/eddl_mnist_conv.cpp)
target_link_libraries(eddl_mnist_conv PUBLIC EDDL::eddl)

add_executable(eddl_mnist_conv_BN eddl/eddl_mnist_conv_BN.cpp)
target_link_libraries(eddl_mnist_conv_BN PUBLIC EDDL::eddl)


