# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/hgh/projects/webrtc_encoding/libdatachannel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hgh/projects/webrtc_encoding/libdatachannel/build

# Include any dependencies generated for this target.
include examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/depend.make

# Include the progress variables for this target.
include examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/progress.make

# Include the compile flags for this target's objects.
include examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/flags.make

examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.o: examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/flags.make
examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.o: ../examples/copy-paste-capi/answerer.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hgh/projects/webrtc_encoding/libdatachannel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.o"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.o   -c /home/hgh/projects/webrtc_encoding/libdatachannel/examples/copy-paste-capi/answerer.c

examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.i"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hgh/projects/webrtc_encoding/libdatachannel/examples/copy-paste-capi/answerer.c > CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.i

examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.s"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hgh/projects/webrtc_encoding/libdatachannel/examples/copy-paste-capi/answerer.c -o CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.s

# Object files for target datachannel-copy-paste-capi-answerer
datachannel__copy__paste__capi__answerer_OBJECTS = \
"CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.o"

# External object files for target datachannel-copy-paste-capi-answerer
datachannel__copy__paste__capi__answerer_EXTERNAL_OBJECTS =

bin/answerer: examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/answerer.c.o
bin/answerer: examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/build.make
bin/answerer: lib/libdatachannel.so.0.20.1
bin/answerer: examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hgh/projects/webrtc_encoding/libdatachannel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../bin/answerer"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/datachannel-copy-paste-capi-answerer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/build: bin/answerer

.PHONY : examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/build

examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/clean:
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi && $(CMAKE_COMMAND) -P CMakeFiles/datachannel-copy-paste-capi-answerer.dir/cmake_clean.cmake
.PHONY : examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/clean

examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/depend:
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hgh/projects/webrtc_encoding/libdatachannel /home/hgh/projects/webrtc_encoding/libdatachannel/examples/copy-paste-capi /home/hgh/projects/webrtc_encoding/libdatachannel/build /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/copy-paste-capi/CMakeFiles/datachannel-copy-paste-capi-answerer.dir/depend
