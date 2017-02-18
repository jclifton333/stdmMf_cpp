find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED system thread filesystem)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  network.proto adaptTestData.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(njm_cpp)
add_library(stdmMf network.cpp system.cpp states.cpp

  ## model
  model.cpp

  ## inf state models
  infStateModel.cpp infStateNoSoModel.cpp infStateOrSoModel.cpp
  infStateXorSoModel.cpp infStateSepSoModel.cpp

  ## inf shield state models
  infShieldStateModel.cpp infShieldStateNoImNoSoModel.cpp
  infShieldStatePosImNoSoModel.cpp

  ## agents
  agent.cpp noTrtAgent.cpp randomAgent.cpp proximalAgent.cpp myopicAgent.cpp
  epsAgent.cpp sweepAgent.cpp sweepAgentSlow.cpp
  vfnMaxSimPerturbAgent.cpp brMinSimPerturbAgent.cpp
  vfnBrAdaptSimPerturbAgent.cpp vfnBrStartSimPerturbAgent.cpp

  ## features
  networkRunFeatures.cpp networkRunSymFeatures.cpp

  ## optimization
  objFns.cpp

  ## extra
  ${PROTO_SRCS})
target_link_libraries(stdmMf njm_cpp)
