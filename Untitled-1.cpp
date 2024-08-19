#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Global variables
cv::Mat frame;
cv::Point cursor_pos(-1, -1);
cv::CascadeClassifier face_cascade;
const int ROI_SIZE = 200; // Size of the region of interest around the cursor

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_MOUSEMOVE)
    {
        cursor_pos = cv::Point(x, y);
    }
}

int main()
{
    // Load the pre-trained face cascade
    if (!face_cascade.load("haarcascade_frontalface_default.xml"))
    {
        std::cout << "Error loading face cascade\n";
        return -1;
    }

    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Error opening video capture\n";
        return -1;
    }

    // Create a window and set mouse callback
    cv::namedWindow("Object Detection");
    cv::setMouseCallback("Object Detection", onMouse, nullptr);

    while (true)
    {
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "No captured frame -- Break!\n";
            break;
        }

        // Define region of interest around cursor
        cv::Rect roi;
        if (cursor_pos.x >= 0 && cursor_pos.y >= 0)
        {
            int x = std::max(0, cursor_pos.x - ROI_SIZE / 2);
            int y = std::max(0, cursor_pos.y - ROI_SIZE / 2);
            int width = std::min(ROI_SIZE, frame.cols - x);
            int height = std::min(ROI_SIZE, frame.rows - y);
            roi = cv::Rect(x, y, width, height);

            // Detect faces in ROI
            std::vector<cv::Rect> faces;
            cv::Mat roi_gray;
            cv::cvtColor(frame(roi), roi_gray, cv::COLOR_BGR2GRAY);
            face_cascade.detectMultiScale(roi_gray, faces, 1.1, 3, 0, cv::Size(30, 30));

            // Draw rectangle and label for each detected face
            for (const auto& face : faces)
            {
                cv::rectangle(frame, cv::Rect(roi.x + face.x, roi.y + face.y, face.width, face.height), cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, "Face", cv::Point(roi.x + face.x, roi.y + face.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
            }

            // Draw ROI rectangle
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
        }

        // Display the resulting frame
        cv::imshow("Object Detection", frame);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q')
            break;
    }

    // Release the capture and destroy windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}