find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED system thread)
find_package(GSL REQUIRED)

include(GetGitRevisionDescription)
git_describe(GIT_DESCRIBE_VAR)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/git_description.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/git_description.cpp" @ONLY)

include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  network.proto)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_library(stdmMf network.cpp random.cpp system.cpp utilities.cpp
  noCovEdgeModel.cpp model.cpp objFns.cpp

  ## agents
  agent.cpp randomAgent.cpp proximalAgent.cpp myopicAgent.cpp
  epsAgent.cpp sweepAgent.cpp sweepAgentSlow.cpp
  vfnMaxSimPerturbAgent.cpp brMinSimPerturbAgent.cpp

  ## features
  networkRunFeatures.cpp

  ## optimization
  optim.cpp simPerturb.cpp

  ## extra
  pool.cpp experiment.cpp
  ${PROTO_SRCS} "${CMAKE_CURRENT_BINARY_DIR}/git_description.cpp")


add_executable(vfnMaxExperiment vfnMaxExperiment.cpp)
target_link_libraries(vfnMaxExperiment stdmMf ${Boost_LIBRARIES}
  ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSL_LIBRARIES})

add_executable(brMinExperiment brMinExperiment.cpp)
target_link_libraries(brMinExperiment stdmMf ${Boost_LIBRARIES}
  ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSL_LIBRARIES})

add_executable(timeSweepAgent timeSweepAgent.cpp)
target_link_libraries(timeSweepAgent stdmMf ${Boost_LIBRARIES}
  ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSL_LIBRARIES})
