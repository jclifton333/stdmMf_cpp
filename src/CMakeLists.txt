find_package(Glog)
find_package(Protobuf REQUIRED)

include(GetGitRevisionDescription)
git_describe(GIT_DESCRIBE_VAR)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/git_description.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/git_description.cpp" @ONLY)

include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  network.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_library(stdmMf network.cpp random.cpp system.cpp utilities.cpp
  noCovEdgeModel.cpp model.cpp runner.cpp

  ## agents
  agent.cpp randomAgent.cpp proximalAgent.cpp myopicAgent.cpp
  epsAgent.cpp sweepAgent.cpp vfnMaxSimPerturbAgent.cpp

  ## features
  networkRunFeatures.cpp

  ## optimization
  optim.cpp simPerturb.cpp

  ## extra
  ${PROTO_SRCS} "${CMAKE_CURRENT_BINARY_DIR}/git_description.cpp")
