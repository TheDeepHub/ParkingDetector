#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// Helper function to preprocess the ROI
cv::Mat preprocess_for_prediction(const cv::Mat& roi, const cv::Size& target_size) {
    cv::Mat roi_resized, roi_normalized;
    cv::resize(roi, roi_resized, target_size);
    roi_resized.convertTo(roi_normalized, CV_32F, 1.0 / 255.0);
    return roi_normalized;
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Specify the model path
    const wchar_t* model_path = L"path_to_your_model.onnx";

    // Create a session for the ONNX model
    Ort::Session session(env, model_path, session_options);

    // Input and output node names (adjust according to your model)
    std::vector<const char*> input_node_names = {"input_node_name"};
    std::vector<const char*> output_node_names = {"output_node_name"};

    // Load the mask
    cv::Mat mask = cv::imread("path_to_mask_image", cv::IMREAD_GRAYSCALE);

    // Video capture
    cv::VideoCapture video_capture("path_to_video_file");
    cv::Mat frame;

    while (true) {
        video_capture >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to grab a frame\n";
            break;
        }

        // Preprocess frame and predict (you need to implement this part)
        // cv::Mat preprocessed_frame = preprocess_for_prediction(frame, cv::Size(target_width, target_height));

        // Display the frame
        cv::imshow("Video with Parking Slot Occupancy", frame);
        if (cv::waitKey(25) == 'q') {
            break;
        }
    }

    // Clean up
    video_capture.release();
    cv::destroyAllWindows();

    return 0;
}

  
