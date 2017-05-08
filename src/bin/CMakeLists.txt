find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED system thread filesystem)
find_package(GSL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
# set(Caffe_DIR "~/tools/share/Caffe")
# find_package(Caffe REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_BINARY_DIR}/src/main" "${CMAKE_SOURCE_DIR}/src/main")


SET(bin_file_names
  ## experiments
  vfnMaxExperiment_infState
  brMinExperiment_infState
  vfnBrAdaptExperiment_infState

  brMinExperiment_infShieldState

  ## sims
  run_infState
  run_infShieldState


  ## adaptive testing
  # adaptTestData_infState
  # adaptTestLikelihood_infState
  # adaptTestMaxLikelihood
  adaptTestData_infShieldState
  adaptTestMaxLikelihood_infShieldState

  ## extra
  inspectBrFn
  orthVecBrFn
  timeSweepAgent
  timeBrMinAgent
  gen_tuples
  # fit_nn
  networksToCsv
  # sandbox
  )

foreach(bin_file ${bin_file_names})
  add_executable(${bin_file} ${bin_file}.cpp)
  target_link_libraries(${bin_file} stdmMf ${Boost_LIBRARIES}
    ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSL_LIBRARIES}
    ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} # ${Caffe_LIBRARIES}
    )

  if (COMPILE_WITH_WARNINGS)
    target_compile_options(${bin_file} PRIVATE -Wall -Wextra)
  endif()
endforeach(bin_file)
