find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED system thread filesystem)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_BINARY_DIR}/src/main" "${CMAKE_SOURCE_DIR}/src/main")

SET(bin_file_names
  vfnMaxExperiment
  brMinExperiment
  vfnBrAdaptExperiment
  timeSweepAgent
  timeBrMinAgent
  run
  adaptTestData
  adaptTestLikelihood
  sandbox)

foreach(bin_file ${bin_file_names})
  add_executable(${bin_file} ${bin_file}.cpp)
  target_link_libraries(${bin_file} stdmMf ${Boost_LIBRARIES}
    ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSL_LIBRARIES}
    ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
endforeach(bin_file)
