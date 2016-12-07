find_package(Glog)
find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS stdmMf.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(stdmMf network.cpp random.cpp system.cpp ${PROTO_SRCS})
