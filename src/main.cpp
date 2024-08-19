#include "object_detector.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

int main(int argc, char* argv[]) {
    try {
        std::cout << "Starting YOLOv3 Object Detector..." << std::endl;
        
        ObjectDetector detector;
        
        std::cout << "Initializing detector..." << std::endl;
        if (!detector.initialize(
            "C:/Users/osyed/OneDrive/Desktop/ObjectDetectionUsingC++/yolov3.cfg",
            "C:/Users/osyed/OneDrive/Desktop/ObjectDetectionUsingC++/yolov3.weights",
            "C:/Users/osyed/OneDrive/Desktop/ObjectDetectionUsingC++/coco.names")) {
            std::cerr << "Failed to initialize the object detector." << std::endl;
            return -1;
        }
        
        if (argc > 1) {
            std::string input = argv[1];
            if (input.find(".jpg") != std::string::npos || input.find(".png") != std::string::npos) {
                std::cout << "Detecting objects in image: " << input << std::endl;
                if (!detector.detectImage(input)) {
                    std::cerr << "Error detecting objects in the image." << std::endl;
                    return -1;
                }
            } else if (input.find(".mp4") != std::string::npos || input.find(".avi") != std::string::npos) {
                std::cout << "Detecting objects in video: " << input << std::endl;
                if (!detector.detectVideo(input)) {
                    std::cerr << "Error detecting objects in the video." << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "Unsupported file format. Please use .jpg, .png, .mp4, or .avi" << std::endl;
                return -1;
            }
        } else {
            std::cout << "Running detector on webcam..." << std::endl;
            if (!detector.run()) {
                std::cerr << "Error occurred while running the object detector." << std::endl;
                return -1;
            }
        }
        
        std::cout << "Detector finished." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
    }

    return 0;
}