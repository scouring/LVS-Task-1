cmake_minimum_required(VERSION 3.16)
project(yolo_inference VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Release build by default ────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgproc videoio)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV include : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs    : ${OpenCV_LIBS}")

# ─── Executable ───────────────────────────────────────────────────────────────
add_executable(yolo_inference src/yolo_inference.cpp)

target_include_directories(yolo_inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_inference PRIVATE ${OpenCV_LIBS})

# ─── Compiler flags ───────────────────────────────────────────────────────────
target_compile_options(yolo_inference PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3 -march=native>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# ─── Install ─────────────────────────────────────────────────────────────────
install(TARGETS yolo_inference DESTINATION bin)
