#include "object_detector.hpp"
#include <iostream>
#include <fstream>

ObjectDetector::ObjectDetector() {}

ObjectDetector::~ObjectDetector() {
    cap.release();
    cv::destroyAllWindows();
}

bool ObjectDetector::initialize(const std::string& modelConfig, const std::string& modelWeights, const std::string& classesFile) {
    // Load the network
    net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open()) {
        std::cerr << "Error opening classes file: " << classesFile << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }

    return true;
}

bool ObjectDetector::run() {
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return false;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "No captured frame -- Break!" << std::endl;
            break;
        }

        detectAndDraw(frame);

        cv::imshow(WINDOW_NAME, frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    return true;
}

bool ObjectDetector::detectImage(const std::string& imagePath) {
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "Error reading image: " << imagePath << std::endl;
        return false;
    }

    detectAndDraw(frame);

    cv::imshow(WINDOW_NAME, frame);
    cv::waitKey(0);

    return true;
}

bool ObjectDetector::detectVideo(const std::string& videoPath) {
    if (videoPath.empty()) {
        return run(); // Use webcam
    }

    cap.open(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return false;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "No captured frame -- Break!" << std::endl;
            break;
        }

        detectAndDraw(frame);

        cv::imshow(WINDOW_NAME, frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    return true;
}

void ObjectDetector::detectAndDraw(cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
    
    net.setInput(blob);
    
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("%.2f", confidences[idx]);
        if (!classes.empty()) {
            CV_Assert(classIds[idx] < (int)classes.size());
            label = classes[classIds[idx]] + ": " + label;
        }
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(frame, label, cv::Point(box.x, box.y - labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
}

std::vector<std::string> ObjectDetector::getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}