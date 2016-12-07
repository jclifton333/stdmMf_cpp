if(__get_git_revision_description)
	return()
endif()
set(__get_git_revision_description YES)


function(git_describe _var)
	if(NOT GIT_FOUND)
		find_package(Git QUIET)
	endif()
  if(NOT GIT_FOUND)
		set(${_var} "GIT-NOTFOUND" PARENT_SCOPE)
		return()
  endif()

  execute_process(COMMAND
		"${GIT_EXECUTABLE}"
    "describe"
    "--always"
    "--dirty"
    WORKING_DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}"
		RESULT_VARIABLE
		res
		OUTPUT_VARIABLE
		out
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(NOT res EQUAL 0)
		set(out "${out}-${res}-NOTFOUND")
	endif()

	set(${_var} "${out}" PARENT_SCOPE)
endfunction()
