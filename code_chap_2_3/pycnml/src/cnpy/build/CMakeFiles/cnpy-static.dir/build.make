# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build

# Include any dependencies generated for this target.
include CMakeFiles/cnpy-static.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cnpy-static.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cnpy-static.dir/flags.make

CMakeFiles/cnpy-static.dir/cnpy.cpp.o: CMakeFiles/cnpy-static.dir/flags.make
CMakeFiles/cnpy-static.dir/cnpy.cpp.o: ../cnpy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cnpy-static.dir/cnpy.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cnpy-static.dir/cnpy.cpp.o -c /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/cnpy.cpp

CMakeFiles/cnpy-static.dir/cnpy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnpy-static.dir/cnpy.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/cnpy.cpp > CMakeFiles/cnpy-static.dir/cnpy.cpp.i

CMakeFiles/cnpy-static.dir/cnpy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnpy-static.dir/cnpy.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/cnpy.cpp -o CMakeFiles/cnpy-static.dir/cnpy.cpp.s

CMakeFiles/cnpy-static.dir/cnpy.cpp.o.requires:

.PHONY : CMakeFiles/cnpy-static.dir/cnpy.cpp.o.requires

CMakeFiles/cnpy-static.dir/cnpy.cpp.o.provides: CMakeFiles/cnpy-static.dir/cnpy.cpp.o.requires
	$(MAKE) -f CMakeFiles/cnpy-static.dir/build.make CMakeFiles/cnpy-static.dir/cnpy.cpp.o.provides.build
.PHONY : CMakeFiles/cnpy-static.dir/cnpy.cpp.o.provides

CMakeFiles/cnpy-static.dir/cnpy.cpp.o.provides.build: CMakeFiles/cnpy-static.dir/cnpy.cpp.o


# Object files for target cnpy-static
cnpy__static_OBJECTS = \
"CMakeFiles/cnpy-static.dir/cnpy.cpp.o"

# External object files for target cnpy-static
cnpy__static_EXTERNAL_OBJECTS =

libcnpy.a: CMakeFiles/cnpy-static.dir/cnpy.cpp.o
libcnpy.a: CMakeFiles/cnpy-static.dir/build.make
libcnpy.a: CMakeFiles/cnpy-static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcnpy.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cnpy-static.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnpy-static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cnpy-static.dir/build: libcnpy.a

.PHONY : CMakeFiles/cnpy-static.dir/build

CMakeFiles/cnpy-static.dir/requires: CMakeFiles/cnpy-static.dir/cnpy.cpp.o.requires

.PHONY : CMakeFiles/cnpy-static.dir/requires

CMakeFiles/cnpy-static.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cnpy-static.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cnpy-static.dir/clean

CMakeFiles/cnpy-static.dir/depend:
	cd /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build /home/Cambricon-Test/shiyan_ch2_3_0610/pycnml/src/cnpy/build/CMakeFiles/cnpy-static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cnpy-static.dir/depend

