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
  run_ebola
  run_ebola_extended

  run_infShieldState_01_18
  run_infShieldState_02_18
  run_infShieldState_03_18
  run_infShieldState_04_18
  run_infShieldState_05_18
  run_infShieldState_06_18
  run_infShieldState_07_18
  run_infShieldState_08_18
  run_infShieldState_09_18
  run_infShieldState_10_18
  run_infShieldState_11_18
  run_infShieldState_12_18
  run_infShieldState_13_18
  run_infShieldState_14_18
  run_infShieldState_15_18
  run_infShieldState_16_18
  run_infShieldState_17_18
  run_infShieldState_18_18

  run_infShieldState_random

  run_infShieldState_extended

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
  networksToFile
  sandbox
  fit_ebola
  fit_ebola_extended
  fit_run_ebola
  vfn_par_history
  )

foreach(bin_file ${bin_file_names})
  add_executable(${bin_file} ${bin_file}.cpp)
  target_link_libraries(${bin_file} stdmMf ${Boost_LIBRARIES}
    ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GSL_LIBRARIES}
    ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} # ${Caffe_LIBRARIES}
    )

  if (${bin_file} MATCHES "^run_infShieldState")
    target_sources(${bin_file} PRIVATE
      "${CMAKE_SOURCE_DIR}/src/main/run_infShieldState_helper.cpp")
  endif()

  if (COMPILE_WITH_WARNINGS)
    target_compile_options(${bin_file} PRIVATE -Wall -Wextra)
  endif()
endforeach(bin_file)
