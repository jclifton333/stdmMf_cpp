enable_testing()

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED system thread)

include_directories(${GTEST_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/build/src)

SET(test_src
  test_network
  test_noCovEdgeModel
  test_noCovEdgeModelMaxSo
  test_utilities
  test_networkRunFeatures
  test_sweepAgent
  test_simPerturb
  test_pool
  test_experiment)

foreach(test_name ${test_src})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES} stdmMf
    ${CMAKE_THREAD_LIBS_INIT} ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES}
    ${GSL_LIBRARIES} ${Boost_LIBRARIES})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach(test_name)
