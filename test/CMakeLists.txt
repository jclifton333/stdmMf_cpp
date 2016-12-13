enable_testing()

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(GSL REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/build/src)

add_executable(test_network test_network.cpp)
target_link_libraries(test_network ${GTEST_BOTH_LIBRARIES} stdmMf
  ${CMAKE_THREAD_LIBS_INIT} ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES})
add_test(NAME test_network COMMAND test_network)

add_executable(test_noCovEdgeModel test_noCovEdgeModel.cpp)
target_link_libraries(test_noCovEdgeModel ${GTEST_BOTH_LIBRARIES} stdmMf
  ${CMAKE_THREAD_LIBS_INIT} ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES}
  ${GSL_LIBRARIES})
add_test(NAME test_noCovEdgeModel COMMAND test_noCovEdgeModel)

add_executable(test_utilities test_utilities.cpp)
target_link_libraries(test_utilities ${GTEST_BOTH_LIBRARIES} stdmMf
  ${CMAKE_THREAD_LIBS_INIT} ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES}
  ${GSL_LIBRARIES})
add_test(NAME test_utilities COMMAND test_utilities)

add_executable(test_networkRunFeatures test_networkRunFeatures.cpp)
target_link_libraries(test_networkRunFeatures ${GTEST_BOTH_LIBRARIES} stdmMf
  ${CMAKE_THREAD_LIBS_INIT} ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES}
  ${GSL_LIBRARIES})
add_test(NAME test_networkRunFeatures COMMAND test_networkRunFeatures)
