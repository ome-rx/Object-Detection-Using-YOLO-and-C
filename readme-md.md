# YOLOv3 Object Detection Project

This project implements a YOLOv3-based object detection system using C++ and OpenCV. It can perform object detection on images, videos, and real-time webcam feeds.

## Prerequisites

- C++ compiler (supporting C++14 or later)
- CMake (version 3.10 or later)
- OpenCV (version 4.x)
- YOLO v3 weights, configuration, and class names files

## Project Structure

```
ObjectDetectionUsingC++/
│
├── src/
│   ├── main.cpp
│   └── object_detector.cpp
│
├── include/
│   └── object_detector.hpp
│
├── CMakeLists.txt
├── yolov3.cfg
├── yolov3.weights
├── coco.names
└── README.md
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ObjectDetectionUsingC++.git
   cd ObjectDetectionUsingC++
   ```

2. Download the YOLOv3 weights file (`yolov3.weights`) from the official YOLO website or GitHub repository and place it in the project root directory.

3. Ensure `yolov3.cfg` and `coco.names` are in the project root directory.

4. Create a build directory and compile the project:
   ```
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```

## Usage

Run the executable from the build directory:

```
./YOLOv3ObjectDetection
```

By default, this will start object detection using your webcam. To use a specific image or video file, provide the file path as an argument:

```
./YOLOv3ObjectDetection path/to/your/image.jpg
```
or
```
./YOLOv3ObjectDetection path/to/your/video.mp4
```

## Configuration

You can adjust the following parameters in `include/object_detector.hpp`:

- `confThreshold`: Confidence threshold for detecting objects (default: 0.5f)
- `nmsThreshold`: Non-maximum suppression threshold (default: 0.4f)
- `inpWidth` and `inpHeight`: Input width and height for the network (default: 416)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
