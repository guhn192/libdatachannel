#include "window.hpp"

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"

#include <opencv2/opencv.hpp>
#include <cstdint> // For uint16_t type
#include <cmath>
#include <chrono> // For high_resolution_clock


const double MAX_RANGE = 2000.0;
const double FRINGE_PITCH = 32.0;
cv::Mat MWD_decode(const cv::Mat& I, double Z_RNG, double Z_MIN) {
    CV_Assert(I.type() == CV_8UC3); // Ensure the input image is of type CV_8UC3

    int width = I.cols;
    int height = I.rows;

    cv::Mat phaseRg = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat phaseB = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat z = cv::Mat::zeros(height, width, CV_32F);

    // Convert to float and scale from 0-255 to 0-1
    cv::Mat I_float;
    I.convertTo(I_float, CV_32FC3, 1.0 / 255);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3f pixel = I_float.at<cv::Vec3f>(y, x);
            float R = pixel[0] - 0.5;
            float G = pixel[1] - 0.5;
            float B = pixel[2];

            // Calculate phase for RG and B channels
            phaseRg.at<float>(y, x) = atan2(R, G);
            phaseB.at<float>(y, x) = B * 2 * M_PI;

            // Reconstruct depth from phase information
            float k = (phaseB.at<float>(y, x) * Z_RNG / FRINGE_PITCH - phaseRg.at<float>(y, x)) / (2 * M_PI);
            float phase = phaseRg.at<float>(y, x) + 2 * M_PI * round(k);
            float zValue = Z_MIN + phase * FRINGE_PITCH / (2 * M_PI);
	    // if(zValue > MAX_RANGE) {
	    //	    continue;
	    // }
            z.at<float>(y, x) = zValue;
        }
    }

    return z;
}

std::tuple<cv::Mat, double, double> MWD_encode(cv::Mat& depthMap) {
    CV_Assert(depthMap.type() == CV_32F);

    //cv::Mat z(height, width, CV_16U, depthData);
    //cv::Mat zFloat; // Convert to float for mathematical operation
    //z.convertTo(zFloat, CV_32F);
    //z = zFloat;
    double Z_MIN, Z_RNG;
    cv::minMaxLoc(depthMap, &Z_MIN, &Z_RNG);
    // Z_RNG = Z_RNG - Z_MIN + 1;

    // std::cout << Z_RNG << " / " << Z_MIN << std::endl;

    cv::Mat I(depthMap.size(), CV_32FC3); // Assuming z is a single-channel, floating-point matrix
    cv::Mat phase = cv::Mat::zeros(depthMap.size(), CV_32F);

    for(int y = 0; y < depthMap.rows; ++y) {
        for(int x = 0; x < depthMap.cols; ++x) {
            float depthValue = depthMap.at<float>(y, x);
	    if (depthValue > MAX_RANGE || std::isnan(depthValue) || depthValue==0) {
		    I.at<cv::Vec3f>(y,x)[0] = NAN;
		    I.at<cv::Vec3f>(y,x)[1] = NAN;
		    I.at<cv::Vec3f>(y,x)[2] = NAN;
		    continue;
	    }

            I.at<cv::Vec3f>(y, x)[0] = 0.5 * (1.0 + sin(2 * M_PI * (depthValue - Z_MIN) / FRINGE_PITCH));
            I.at<cv::Vec3f>(y, x)[1] = 0.5 * (1.0 + cos(2 * M_PI * (depthValue - Z_MIN) / FRINGE_PITCH));

            float Ibsin = 0.5 * (1.0 + sin(2 * M_PI * fmod(depthValue - Z_MIN, Z_RNG) / Z_RNG));
            float Ibcos = 0.5 * (1.0 + cos(2 * M_PI * fmod(depthValue - Z_MIN, Z_RNG) / Z_RNG));

            phase.at<float>(y, x) = fmod(atan2(Ibsin - 0.5, Ibcos - 0.5), 2 * M_PI);
            I.at<cv::Vec3f>(y, x)[2] = phase.at<float>(y, x) / (2 * M_PI);
        }
    }

    // Handle NaN values if necessary, similar to MATLAB's mask handling
    I.convertTo(I, CV_8UC3, 255.0);

    return std::make_tuple(I, Z_RNG, Z_MIN); // Returns the encoded image as a 3-channel float matrix
}

cv::Mat scaleDepthMap(uint16_t* depthData, int width, int height, float scale) {
        cv::Mat z(height, width, CV_16U, depthData);
        cv::Mat zScaled(height, width, CV_32F);
        for(int y = 0; y < z.rows; ++y) {
            for(int x = 0; x < z.cols; ++x) {
                 float depthValue = z.at<uint16_t>(y, x) * scale;
                 if(depthValue > MAX_RANGE || depthValue < 600 || std::isnan(depthValue)) {
                         depthValue = NAN;
                 }
                 zScaled.at<float>(y,x) = depthValue;
            }
        }
        return zScaled;
}


cv::Mat depthMapToHeatMap(cv::Mat depthMap) {
	//cv::Mat depthMap(height, width, CV_16UC1, depthData);
        double minDepth, maxDepth;
        cv::minMaxLoc(depthMap, &minDepth, &maxDepth);

        // Normalize the depth map to 0-255
        cv::Mat normalizedDepthMap;
        depthMap.convertTo(normalizedDepthMap, CV_8UC1, 255.0 / (maxDepth-minDepth), -minDepth*255.0/(maxDepth-minDepth));

        // Apply a colormap for visualization
        cv::Mat depthHeatMap;
        cv::applyColorMap(normalizedDepthMap, depthHeatMap, cv::COLORMAP_JET);
	return depthHeatMap;
}

// Function to calculate RMSE given an accuracy map, considering NaN values
double calculateRMSE(const cv::Mat& accuracyMap) {
    // Ensure the accuracy map is of floating point type for accurate calculations
    CV_Assert(accuracyMap.type() == CV_32F || accuracyMap.type() == CV_64F);

    // Create a mask where true indicates that the value is not NaN (i.e., it's a number)
    cv::Mat mask = cv::Mat(accuracyMap == accuracyMap); // NaN will not be equal to itself

    // Calculate the squared differences, using the mask to ignore NaN values
    cv::Mat squaredDifferences;
    cv::pow(accuracyMap, 2, squaredDifferences);
    squaredDifferences.setTo(0, ~mask); // Set squared differences to 0 where mask is false (where original values were NaN)

    // Compute the mean of the squared differences using the mask to consider only valid (non-NaN) values
    cv::Scalar meanSquaredError = cv::mean(squaredDifferences, mask);

    // Take the square root of the mean squared error to get RMSE
    double rmse = std::sqrt(meanSquaredError[0]);

    return rmse;
}

int main(int argc, char **argv) try {
    // Create a pipeline with default device
    ob::Pipeline pipe;

    // Get all stream profiles of the depth camera, including stream resolution, frame rate, and frame format
    auto profiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);

    std::shared_ptr<ob::VideoStreamProfile> depthProfile = nullptr;
    try {
        // Find the corresponding profile according to the specified format, first look for the y16 format
        depthProfile = profiles->getVideoStreamProfile(800, OB_HEIGHT_ANY, OB_FORMAT_Y16, 30);
    }
    catch(ob::Error &e) {
        // If the specified format is not found, search for the default profile to open the stream
        depthProfile = std::const_pointer_cast<ob::StreamProfile>(profiles->getProfile(OB_PROFILE_DEFAULT))->as<ob::VideoStreamProfile>();
    }

    // By creating config to configure which streams to enable or disable for the pipeline, here the depth stream will be enabled
    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();
    config->enableStream(depthProfile);

    // Start the pipeline with config
    pipe.start(config);

    // Calculate FPS
    auto lastTime = std::chrono::high_resolution_clock::now();
    double fpsOriginal = 0.0, fpsDecoded = 0.0;

    while(true) {

	// Inside the while loop, right before processing the depth map
        auto startTime = std::chrono::high_resolution_clock::now();
        
	// Wait for up to 100ms for a frameset in blocking mode.
        auto frameSet = pipe.waitForFrames(1000);
        if(frameSet == nullptr) {
            continue;
        }

        auto depthFrame = frameSet->depthFrame();

	// After processing the original depth map
        auto endTimeOriginal = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedOriginal = endTimeOriginal - startTime;
        //fpsOriginal = 1.0 / elapsedOriginal.count();
        //std::cout << "FPS (Original Depth Map): " << fpsOriginal << std::endl;
 	

        // for Y16 format depth frame, print the distance of the center pixel every 30 frames
        // if(depthFrame->index() % 30 == 0 && depthFrame->format() == OB_FORMAT_Y16) {
        uint32_t  width  = depthFrame->width();
        uint32_t  height = depthFrame->height();
        float     scale  = depthFrame->getValueScale();
        uint16_t *data   = (uint16_t *)depthFrame->data();
    
        // pixel value multiplied by scale is the actual distance value in millimeters
        float centerDistance = data[width * height / 2 + width / 2] * scale;
    
	// Step 1 : Scale depth to mm
	cv::Mat depthScaled = scaleDepthMap(data, width, height, scale);

	// Step 2 : Encode depth using MWD
        auto [encodedImage, Z_RNG, Z_MIN] = MWD_encode(depthScaled);


	//cv::Mat depthMap(height, width, CV_16UC1, data);
	//cv::Mat depthHeatMap = depthMapToHeatMap(depthMap);
	// Step 3 : Decode depth using MWD
	cv::Mat decodedDepth = MWD_decode(encodedImage, Z_RNG, Z_MIN);

	// After decoding the depth map
        auto endTimeDecoded = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedDecoded = endTimeDecoded - startTime;
        fpsDecoded = 1.0 / elapsedDecoded.count();
        std::cout << "FPS (Decoded Depth Map): " << fpsDecoded << std::endl;

	// Get heatmap of decoded depth map to visualize
	cv::Mat decodedDepthHeatMap = depthMapToHeatMap(decodedDepth);

        //cv::imshow("Encoded Viewer", encodedImage);
        //cv::imshow("Depth Viewer", depthHeatmap);
	cv::Mat depthScaledHeatMap = depthMapToHeatMap(depthScaled);

	// Compute the accuracy map
        cv::Mat accuracyMap = cv::abs(depthScaled - decodedDepth);
        cv::Mat accuracyHeatMap = depthMapToHeatMap(accuracyMap);

	// double rmse = calculateRMSE(accuracyMap);
	// std::cout << "RMSE: " << rmse << std::endl;


	// Convert FPS value to string with fixed precision
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << "FPS (Decoded): " << fpsDecoded;
        std::string fpsText = ss.str();

        // Choose a position on the image to display the text (e.g., top left corner)
        cv::Point textPos(30, 30); // 10 pixels from the left, 30 pixels from the top

        // Define font face and scale
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.7;
        int thickness = 2;

        // Get the text size for background rectangle
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(fpsText, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        // Set the background rectangle for better readability
        cv::rectangle(decodedDepthHeatMap, textPos + cv::Point(0, baseline), textPos + cv::Point(textSize.width, -textSize.height), CV_RGB(255,255,255), cv::FILLED);

        // Set the text color (e.g., black)
        cv::Scalar textColor(0, 0, 0);

        // Overlay the text on the image
        cv::putText(decodedDepthHeatMap, fpsText, textPos, fontFace, fontScale, textColor, thickness);


	cv::Mat combinedImage1;
        cv::hconcat(depthScaledHeatMap, encodedImage, combinedImage1); // Horizontally combine images

	cv::Mat combinedImage2;
	cv::hconcat(decodedDepthHeatMap, accuracyHeatMap, combinedImage2);

	cv::Mat combinedImage;
	cv::vconcat(combinedImage1, combinedImage2, combinedImage);

	cv::namedWindow("Viewer", cv::WINDOW_NORMAL); // Create a window that can be resized
	cv::resizeWindow("Viewer", 800,600);
        cv::imshow("Viewer", combinedImage); // Show combined image
	
        cv::waitKey(1);

        // attention: if the distance is 0, it means that the depth camera cannot detect the object（may be out of detection range）
        // std::cout << "Facing an object " << centerDistance << " mm away. " << std::endl;
    }
    // Stop the pipeline
    pipe.stop();

    return 0;
}
catch(ob::Error &e) {
    std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs() << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
    exit(EXIT_FAILURE);
}
