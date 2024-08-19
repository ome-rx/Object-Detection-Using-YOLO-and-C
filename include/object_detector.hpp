#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();

    bool initialize(const std::string& modelConfig, const std::string& modelWeights, const std::string& classesFile);
    bool run();
    bool detectImage(const std::string& imagePath);
    bool detectVideo(const std::string& videoPath = "");

private:
    void detectAndDraw(cv::Mat& frame);
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);

    cv::VideoCapture cap;
    cv::dnn::Net net;
    std::vector<std::string> classes;
    const std::string WINDOW_NAME = "YOLOv3 Object Detection";
    
    // YOLO parameters
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    int inpWidth = 416;        // Width of network's input image
    int inpHeight = 416;       // Height of network's input image
};

#endif // OBJECT_DETECTOR_HPP
