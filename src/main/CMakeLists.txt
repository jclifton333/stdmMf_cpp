find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED system thread filesystem)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)

IF(APPLE)
  include_directories("/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/")
ENDIF(APPLE)

include(GetGitRevisionDescription)
git_describe(GIT_SHA1_VAR)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/projectInfo.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/projectInfo.cpp" @ONLY)

include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  network.proto adaptTestData.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_library(stdmMf network.cpp random.cpp system.cpp utilities.cpp types.cpp

  ## models
  model.cpp noCovEdgeModel.cpp noCovEdgeOrSoModel.cpp noCovEdgeXorSoModel.cpp
  noCovEdgeSepSoModel.cpp

  ## agents
  agent.cpp noTrtAgent.cpp randomAgent.cpp proximalAgent.cpp myopicAgent.cpp
  epsAgent.cpp sweepAgent.cpp sweepAgentSlow.cpp
  vfnMaxSimPerturbAgent.cpp brMinSimPerturbAgent.cpp
  vfnBrAdaptSimPerturbAgent.cpp vfnBrStartSimPerturbAgent.cpp

  ## features
  networkRunFeatures.cpp networkRunSymFeatures.cpp

  ## optimization
  optim.cpp simPerturb.cpp objFns.cpp

  ## extra
  pool.cpp experiment.cpp trapperKeeper.cpp
  ${PROTO_SRCS} "${CMAKE_CURRENT_BINARY_DIR}/projectInfo.cpp")
