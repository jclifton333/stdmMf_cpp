enable_testing()

find_package(Threads REQUIRED)
find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(GSL REQUIRED)
find_package(Boost REQUIRED system thread filesystem)

include_directories(${gtest_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src/main ${CMAKE_BINARY_DIR}/src/main)

SET(test_src
  test_network
  test_noCovEdgeModel
  test_noCovEdgeOrSoModel
  test_noCovEdgeXorSoModel
  test_noCovEdgeSepSoModel
  test_networkRunFeatures
  test_networkRunSymFeatures
  test_sweepAgent)

foreach(test_name ${test_src})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} gtest_main stdmMf
    ${CMAKE_THREAD_LIBS_INIT} ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES}
    ${GSL_LIBRARIES} ${Boost_LIBRARIES})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach(test_name)
