cmake_minimum_required(VERSION 3.10)
project(YOLOv3ObjectDetection)

set(CMAKE_CXX_STANDARD 14)

# Specify the path to OpenCV config if it's not in a standard location
set(OpenCV_DIR "C:/lib/install/opencv/x64/vc16/lib")

find_package(OpenCV REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(YOLOv3ObjectDetection src/main.cpp src/object_detector.cpp)
target_link_libraries(YOLOv3ObjectDetection ${OpenCV_LIBS})

# Copy YOLO files to build directory
file(COPY "${CMAKE_SOURCE_DIR}/yolov3.cfg" DESTINATION "${CMAKE_BINARY_DIR}/")
file(COPY "${CMAKE_SOURCE_DIR}/yolov3.weights" DESTINATION "${CMAKE_BINARY_DIR}/")
file(COPY "${CMAKE_SOURCE_DIR}/coco.names" DESTINATION "${CMAKE_BINARY_DIR}/")