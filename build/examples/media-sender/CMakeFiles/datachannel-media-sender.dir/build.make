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
include examples/media-sender/CMakeFiles/datachannel-media-sender.dir/depend.make

# Include the progress variables for this target.
include examples/media-sender/CMakeFiles/datachannel-media-sender.dir/progress.make

# Include the compile flags for this target's objects.
include examples/media-sender/CMakeFiles/datachannel-media-sender.dir/flags.make

examples/media-sender/CMakeFiles/datachannel-media-sender.dir/main.cpp.o: examples/media-sender/CMakeFiles/datachannel-media-sender.dir/flags.make
examples/media-sender/CMakeFiles/datachannel-media-sender.dir/main.cpp.o: ../examples/media-sender/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hgh/projects/webrtc_encoding/libdatachannel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/media-sender/CMakeFiles/datachannel-media-sender.dir/main.cpp.o"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/datachannel-media-sender.dir/main.cpp.o -c /home/hgh/projects/webrtc_encoding/libdatachannel/examples/media-sender/main.cpp

examples/media-sender/CMakeFiles/datachannel-media-sender.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/datachannel-media-sender.dir/main.cpp.i"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hgh/projects/webrtc_encoding/libdatachannel/examples/media-sender/main.cpp > CMakeFiles/datachannel-media-sender.dir/main.cpp.i

examples/media-sender/CMakeFiles/datachannel-media-sender.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/datachannel-media-sender.dir/main.cpp.s"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hgh/projects/webrtc_encoding/libdatachannel/examples/media-sender/main.cpp -o CMakeFiles/datachannel-media-sender.dir/main.cpp.s

# Object files for target datachannel-media-sender
datachannel__media__sender_OBJECTS = \
"CMakeFiles/datachannel-media-sender.dir/main.cpp.o"

# External object files for target datachannel-media-sender
datachannel__media__sender_EXTERNAL_OBJECTS =

bin/media-sender: examples/media-sender/CMakeFiles/datachannel-media-sender.dir/main.cpp.o
bin/media-sender: examples/media-sender/CMakeFiles/datachannel-media-sender.dir/build.make
bin/media-sender: lib/libdatachannel.so.0.20.1
bin/media-sender: examples/media-sender/CMakeFiles/datachannel-media-sender.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hgh/projects/webrtc_encoding/libdatachannel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/media-sender"
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/datachannel-media-sender.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/media-sender/CMakeFiles/datachannel-media-sender.dir/build: bin/media-sender

.PHONY : examples/media-sender/CMakeFiles/datachannel-media-sender.dir/build

examples/media-sender/CMakeFiles/datachannel-media-sender.dir/clean:
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender && $(CMAKE_COMMAND) -P CMakeFiles/datachannel-media-sender.dir/cmake_clean.cmake
.PHONY : examples/media-sender/CMakeFiles/datachannel-media-sender.dir/clean

examples/media-sender/CMakeFiles/datachannel-media-sender.dir/depend:
	cd /home/hgh/projects/webrtc_encoding/libdatachannel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hgh/projects/webrtc_encoding/libdatachannel /home/hgh/projects/webrtc_encoding/libdatachannel/examples/media-sender /home/hgh/projects/webrtc_encoding/libdatachannel/build /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender /home/hgh/projects/webrtc_encoding/libdatachannel/build/examples/media-sender/CMakeFiles/datachannel-media-sender.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/media-sender/CMakeFiles/datachannel-media-sender.dir/depend
