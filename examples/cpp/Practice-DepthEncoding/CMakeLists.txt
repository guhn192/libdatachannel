#  minimum required cmake version: 3.1.15 support vs2019
cmake_minimum_required(VERSION 3.1.15)
project(OBPracticeDepthEncoding)

if(POLICY CMP0079)
	cmake_policy(SET CMP0079 NEW)
endif()

set(MEDIA_UWP_RESOURCES
	uwp/Logo.png
	uwp/package.appxManifest
	uwp/SmallLogo.png
	uwp/SmallLogo44x44.png
	uwp/SplashScreen.png
	uwp/StoreLogo.png
	uwp/Windows_TemporaryKey.pfx
)

set(CLIENT_SOURCES
	PracticeDepthEncoding.cpp
	parse_cl.cpp
	parse_cl.h
)

set(GETOPT_SOURCES
	getopt.cpp
	getopt.h
)


add_executable(${PROJECT_NAME}
	PracticeDepthEncoding.cpp
  ${CLIENT_SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
)

set_target_properties(${PROJECT_NAME} PROPERTIES
	XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.github.paullouisageneau.libdatachannel.examples.media-sender)


target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${OrbbecSDK_LIBS}
  LibDataChannel::LibDataChannel 
  Threads::Threads 
  nlohmann_json::nlohmann_json
  )

find_package(Threads REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${OrbbecSDK_INCLUDE_DIRS}
  )

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )
