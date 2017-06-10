find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED system thread filesystem)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)
# set(Caffe_DIR "~/tools/share/Caffe")
# find_package(Caffe)

include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  network.proto sim_data.proto)
protobuf_generate_python(PROTO_PY network.proto sim_data.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(njm_cpp)
add_library(stdmMf network.cpp system.cpp states.cpp

  ## model
  model.cpp mixtureModel.cpp

  ## inf state models
  infStateModel.cpp infStateNoSoModel.cpp infStateOrSoModel.cpp
  infStateXorSoModel.cpp infStateSepSoModel.cpp

  ## inf shield state models
  infShieldStateModel.cpp infShieldStateNoImNoSoModel.cpp
  infShieldStatePosImNoSoModel.cpp

  ## ebola state models
  ebolaStateModel.cpp ebolaStateGravityModel.cpp

  ## agents
  agent.cpp noTrtAgent.cpp randomAgent.cpp proximalAgent.cpp myopicAgent.cpp
  epsAgent.cpp sweepAgent.cpp sweepAgentSlow.cpp
  vfnMaxSimPerturbAgent.cpp brMinSimPerturbAgent.cpp
  vfnBrAdaptSimPerturbAgent.cpp vfnBrStartSimPerturbAgent.cpp
  brMinWtdSimPerturbAgent.cpp
  lookupAgent.cpp

  ## features
  features.cpp networkRunFeatures.cpp networkRunSymFeatures.cpp
  finiteQfnFeatures.cpp # finiteQfnNnFeatures.cpp

  ## optimization
  objFns.cpp polValIteration.cpp

  ## extra
  # neuralNetwork.cpp
  ${PROTO_SRCS}
  ${PROTO_PY})

target_link_libraries(stdmMf njm_cpp # ${Caffe_LIBRARIES}
  )


if (COMPILE_WITH_WARNINGS)
  target_compile_options(stdmMf PRIVATE -Wall -Wextra)
endif()
