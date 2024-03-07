/**
 * libdatachannel client example
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 * Copyright (c) 2019 Murat Dogan
 * Copyright (c) 2020 Will Munn
 * Copyright (c) 2020 Nico Chatzi
 * Copyright (c) 2020 Lara Mackey
 * Copyright (c) 2020 Erik Cota-Robles
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "rtc/rtc.hpp"

#include "parse_cl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace std::chrono_literals;
using std::shared_ptr;
using std::weak_ptr;
template <class T> weak_ptr<T> make_weak_ptr(shared_ptr<T> ptr) { return ptr; }

using nlohmann::json;

std::string localId;
std::unordered_map<std::string, shared_ptr<rtc::PeerConnection>> peerConnectionMap;
std::unordered_map<std::string, shared_ptr<rtc::DataChannel>> dataChannelMap;

shared_ptr<rtc::PeerConnection> createPeerConnection(const rtc::Configuration &config,
                                                     weak_ptr<rtc::WebSocket> wws, std::string id);
std::string randomId(size_t length);

#include "window.hpp"

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"

#include <opencv2/opencv.hpp>
#include <cstdint> // For uint16_t type
#include <cmath>
#include <chrono> // For high_resolution_clock
#include <opencv2/imgcodecs.hpp> // Include for imencode


const double MIN_RANGE = 600.0;
const double MAX_RANGE = 1200.0;
const double FRINGE_PITCH = 32.0;

const int ROI_I = 400;
const int ROI_J = 400;
const int N_MIN = 1;
const int N_MAX = 12;
const double alpha = 1.1;
const double beta = 7.0;

cv::Mat getNormalizedDistributedMap(int height, int width) {
    cv::Mat E = cv::Mat(height, width, CV_32F);
    float maxE = 0;

    // Calculate distance map
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float distance = sqrt(pow(i - ROI_I, 2) + pow(j - ROI_J, 2));
            E.at<float>(i, j) = distance;
            if (distance > maxE) maxE = distance;
        }
    }

    // Normalize and adjust based on alpha and beta
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            E.at<float>(i, j) = pow((alpha - (E.at<float>(i, j) / maxE)), beta);
            if (E.at<float>(i, j) > 1) E.at<float>(i, j) = 1;
        }
    }

    // Scale between N_MIN and N_MAX
    E = (E * (N_MAX - N_MIN)) + N_MIN;

    return E;
}

cv::Mat Foveated_decode(const cv::Mat& encodedImage, double Z_RNG, double Z_MIN) {
    CV_Assert(encodedImage.type() == CV_8UC3); // Ensure the encoded image is of type CV_8UC3

    cv::Mat decodedDepthMap = cv::Mat(encodedImage.rows, encodedImage.cols, CV_32F);
    cv::Mat N = getNormalizedDistributedMap(encodedImage.rows, encodedImage.cols);

    for (int y = 0; y < encodedImage.rows; ++y) {
        for (int x = 0; x < encodedImage.cols; ++x) {
            cv::Vec3b pixel = encodedImage.at<cv::Vec3b>(y, x);
            float nValue = N.at<float>(y, x);
	    if(pixel[2] == 0.0f) {
                decodedDepthMap.at<float>(y, x) = NAN;
	        continue;
	    }

            // Decode the depth value from the three channels
            float normValue = pixel[2] / 255.0f; // Normalize the third channel value back to [0, 1]
            float sinValue = (pixel[0] / 255.0f) - 0.5f;
            float cosValue = (pixel[1] / 255.0f) - 0.5f;

            float phaseRg = atan2(sinValue, cosValue);
            // if (phase < 0) phase += 2 * M_PI; // Ensure phase is positive
	    
	    float phaseB = normValue * 2 * M_PI;
	    float k = round( (phaseB * nValue - phaseRg) / (2 * M_PI) );
	    float phase = phaseRg + 2 * M_PI * k;
	    float depthValue = Z_MIN + phase * Z_RNG / (2 * M_PI * nValue);
	    
            decodedDepthMap.at<float>(y, x) = depthValue;
        }
    }

    return decodedDepthMap;
}


std::tuple<cv::Mat, double, double> Foveated_encode(cv::Mat& depthMap) {
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


     // Generate the normalized distribution map
    cv::Mat N = getNormalizedDistributedMap(depthMap.rows, depthMap.cols);


    for(int y = 0; y < depthMap.rows; ++y) {
        for(int x = 0; x < depthMap.cols; ++x) {
            float depthValue = depthMap.at<float>(y, x);
            if (depthValue > MAX_RANGE || std::isnan(depthValue) || depthValue < MIN_RANGE) {
                    I.at<cv::Vec3f>(y,x)[0] = NAN;
                    I.at<cv::Vec3f>(y,x)[1] = NAN;
                    I.at<cv::Vec3f>(y,x)[2] = NAN;
                    continue;
            }

	                float zValue = depthMap.at<float>(y, x);
            float normValue = (zValue - Z_MIN) / Z_RNG;
            float nValue = N.at<float>(y, x);

            // Encode the depth value into the three channels with varying precision
            I.at<cv::Vec3f>(y, x)[0] = 0.5f * (1.0f + std::sin(2 * M_PI * normValue * nValue));
            I.at<cv::Vec3f>(y, x)[1] = 0.5f * (1.0f + std::cos(2 * M_PI * normValue * nValue));
            I.at<cv::Vec3f>(y, x)[2] = normValue;

	    // MWD
            //I.at<cv::Vec3f>(y, x)[0] = 0.5 * (1.0 + sin(2 * M_PI * (depthValue - Z_MIN) / FRINGE_PITCH));
            //I.at<cv::Vec3f>(y, x)[1] = 0.5 * (1.0 + cos(2 * M_PI * (depthValue - Z_MIN) / FRINGE_PITCH));

            //float Ibsin = 0.5 * (1.0 + sin(2 * M_PI * fmod(depthValue - Z_MIN, Z_RNG) / Z_RNG));
            //float Ibcos = 0.5 * (1.0 + cos(2 * M_PI * fmod(depthValue - Z_MIN, Z_RNG) / Z_RNG));

            //phase.at<float>(y, x) = fmod(atan2(Ibsin - 0.5, Ibcos - 0.5), 2 * M_PI);
            //I.at<cv::Vec3f>(y, x)[2] = phase.at<float>(y, x) / (2 * M_PI);
        }
    }

    // Handle NaN values if necessary, similar to MATLAB's mask handling
    I.convertTo(I, CV_8UC3, 255.0);

    return std::make_tuple(I, Z_RNG, Z_MIN); // Returns the encoded image as a 3-channel float matrix
}


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
	    if(pixel[2] == 0.0f) {
                z.at<float>(y, x) = NAN;
                continue;
            }

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
                 if(depthValue > MAX_RANGE || depthValue < MIN_RANGE || std::isnan(depthValue)) {
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

std::tuple<cv::Mat, cv::Mat> depthMapToHeatMapMulti(cv::Mat depthMap1, cv::Mat depthMap2) {
	double minDepth1, maxDepth1, minDepth2, maxDepth2, minDepth, maxDepth;
	cv::minMaxLoc(depthMap1, &minDepth1, &maxDepth1);
	cv::minMaxLoc(depthMap2, &minDepth2, &maxDepth2);
	
	if(minDepth1 > minDepth2) {
		minDepth = minDepth2;
	} else {
		minDepth = minDepth1;
	}

	if(maxDepth1 > maxDepth2) {
		maxDepth = maxDepth1;
	} else {
		maxDepth = maxDepth2;
	}

	// Normalize the depth map to 0-255
	cv::Mat normalizedDepthMap1;
	cv::Mat normalizedDepthMap2;
	depthMap1.convertTo(normalizedDepthMap1, CV_8UC1, 255.0 / (maxDepth-minDepth), -minDepth*255.0/(maxDepth-minDepth));
	depthMap2.convertTo(normalizedDepthMap2, CV_8UC1, 255.0 / (maxDepth-minDepth), -minDepth*255.0/(maxDepth-minDepth));
	cv::Mat depthHeatMap1;
	cv::Mat depthHeatMap2;
	cv::applyColorMap(normalizedDepthMap1, depthHeatMap1, cv::COLORMAP_JET);
	cv::applyColorMap(normalizedDepthMap2, depthHeatMap2, cv::COLORMAP_JET);
	return std::make_tuple(depthHeatMap1, depthHeatMap2);
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
	Cmdline params(argc, argv);

	rtc::InitLogger(rtc::LogLevel::Info);

	rtc::Configuration config;
	std::string stunServer = "";
	if (params.noStun()) {
		std::cout
		    << "No STUN server is configured. Only local hosts and public IP addresses supported."
		    << std::endl;
	} else {
		if (params.stunServer().substr(0, 5).compare("stun:") != 0) {
			stunServer = "stun:";
		}
		stunServer += params.stunServer() + ":" + std::to_string(params.stunPort());
		std::cout << "STUN server is " << stunServer << std::endl;
		config.iceServers.emplace_back(stunServer);
	}

	if (params.udpMux()) {
		std::cout << "ICE UDP mux enabled" << std::endl;
		config.enableIceUdpMux = true;
	}

	localId = randomId(4);
	std::cout << "The local ID is " << localId << std::endl;

	auto ws = std::make_shared<rtc::WebSocket>();

	std::promise<void> wsPromise;
	auto wsFuture = wsPromise.get_future();

	ws->onOpen([&wsPromise]() {
		std::cout << "WebSocket connected, signaling ready" << std::endl;
		wsPromise.set_value();
	});

	ws->onError([&wsPromise](std::string s) {
		std::cout << "WebSocket error" << std::endl;
		wsPromise.set_exception(std::make_exception_ptr(std::runtime_error(s)));
	});

	ws->onClosed([]() { std::cout << "WebSocket closed" << std::endl; });

	ws->onMessage([&config, wws = make_weak_ptr(ws)](auto data) {
		// data holds either std::string or rtc::binary
		if (!std::holds_alternative<std::string>(data))
			return;

		json message = json::parse(std::get<std::string>(data));

		auto it = message.find("id");
		if (it == message.end())
			return;

		auto id = it->get<std::string>();

		it = message.find("type");
		if (it == message.end())
			return;

		auto type = it->get<std::string>();

		shared_ptr<rtc::PeerConnection> pc;
		if (auto jt = peerConnectionMap.find(id); jt != peerConnectionMap.end()) {
			pc = jt->second;
		} else if (type == "offer") {
			std::cout << "Answering to " + id << std::endl;
			pc = createPeerConnection(config, wws, id);
		} else {
			return;
		}

		if (type == "offer" || type == "answer") {
			auto sdp = message["description"].get<std::string>();
			pc->setRemoteDescription(rtc::Description(sdp, type));
		} else if (type == "candidate") {
			auto sdp = message["candidate"].get<std::string>();
			auto mid = message["mid"].get<std::string>();
			pc->addRemoteCandidate(rtc::Candidate(sdp, mid));
		}
	});

	const std::string wsPrefix =
	    params.webSocketServer().find("://") == std::string::npos ? "ws://" : "";
	const std::string url = wsPrefix + params.webSocketServer() + ":" +
	                        std::to_string(params.webSocketPort()) + "/" + localId;

	std::cout << "WebSocket URL is " << url << std::endl;
	ws->open(url);

	std::cout << "Waiting for signaling to be connected..." << std::endl;
	wsFuture.get();

	while (true) {
		std::string id;
		std::cout << "Enter a remote ID to send an offer:" << std::endl;
		std::cin >> id;
		std::cin.ignore();

		if (id.empty())
			break;

		if (id == localId) {
			std::cout << "Invalid remote ID (This is the local ID)" << std::endl;
			continue;
		}

		std::cout << "Offering to " + id << std::endl;
		auto pc = createPeerConnection(config, ws, id);

		// We are the offerer, so create a data channel to initiate the process
		const std::string label = "test";
		std::cout << "Creating DataChannel with label \"" << label << "\"" << std::endl;
		auto dc = pc->createDataChannel(label);

		dc->onOpen([id, wdc = make_weak_ptr(dc)]() {
			std::cout << "DataChannel from " << id << " open" << std::endl;
			if (auto dc = wdc.lock())
				dc->send("Hello from " + localId);
		});

		dc->onClosed([id]() { std::cout << "DataChannel from " << id << " closed" << std::endl; });

		dc->onMessage([id, wdc = make_weak_ptr(dc)](auto data) {
			// data holds either std::string or rtc::binary
			if (std::holds_alternative<std::string>(data))
				std::cout << "Message from " << id << " received: " << std::get<std::string>(data)
				          << std::endl;
			else
				std::cout << "Binary message from " << id
				          << " received, size=" << std::get<rtc::binary>(data).size() << std::endl;
		});

		dataChannelMap.emplace(id, dc);
	}

	std::cout << "Cleaning up..." << std::endl;

	dataChannelMap.clear();
	peerConnectionMap.clear();
	return 0;

} catch (const std::exception &e) {
	std::cout << "Error: " << e.what() << std::endl;
	dataChannelMap.clear();
	peerConnectionMap.clear();
	return -1;
}

// Create and setup a PeerConnection
shared_ptr<rtc::PeerConnection> createPeerConnection(const rtc::Configuration &config,
                                                     weak_ptr<rtc::WebSocket> wws, std::string id) {
	auto pc = std::make_shared<rtc::PeerConnection>(config);

	pc->onStateChange(
	    [](rtc::PeerConnection::State state) { std::cout << "State: " << state << std::endl; });

	pc->onGatheringStateChange([](rtc::PeerConnection::GatheringState state) {
		std::cout << "Gathering State: " << state << std::endl;
	});

	pc->onLocalDescription([wws, id](rtc::Description description) {
		json message = {{"id", id},
		                {"type", description.typeString()},
		                {"description", std::string(description)}};

		if (auto ws = wws.lock())
			ws->send(message.dump());
	});

	pc->onLocalCandidate([wws, id](rtc::Candidate candidate) {
		json message = {{"id", id},
		                {"type", "candidate"},
		                {"candidate", std::string(candidate)},
		                {"mid", candidate.mid()}};

		if (auto ws = wws.lock())
			ws->send(message.dump());
	});

	pc->onDataChannel([id](shared_ptr<rtc::DataChannel> dc) {
		std::cout << "DataChannel from " << id << " received with label \"" << dc->label() << "\""
		          << std::endl;

		dc->onOpen([wdc = make_weak_ptr(dc)]() {
            std::cout << "Data channel open, ready to send images." << std::endl;

            // Example: Send an image every 5 seconds
            std::thread([wdc]() {
                 // Create a pipeline with default device
                ob::Pipeline pipe;
                std::cout << "Create a pipeline with default device" << std::endl;

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
                while (auto dc = wdc.lock()) {
                    if (!dc->isOpen()) break;
                    auto frameSet = pipe.waitForFrames(100);
                    if(frameSet == nullptr) {
                        continue;
                    }

                    auto depthFrame = frameSet->depthFrame();
                    uint32_t  width  = depthFrame->width();
                    uint32_t  height = depthFrame->height();
                    float     scale  = depthFrame->getValueScale();
                    uint16_t *data   = (uint16_t *)depthFrame->data();

                    float centerDistance = data[width * height / 2 + width / 2] * scale;
                    // std::cout << "centerDistance : " << centerDistance << std::endl;
                    
                    // Step 1 : Scale depth to mm
	                cv::Mat depthScaled = scaleDepthMap(data, width, height, scale);

                    // Step 2 : Encode depth using MWD
                    auto [MWDEncodedImage, MWD_Z_RNG, MWD_Z_MIN] = MWD_encode(depthScaled);
                    // Encode MWDEncodedImage to JPEG
                    std::vector<uchar> buf;
                    cv::imencode(".jpg", MWDEncodedImage, buf);

                    std::vector<std::byte> encodedImage;
                    encodedImage.reserve(buf.size());
                    for (auto& b : buf) {
                        encodedImage.push_back(static_cast<std::byte>(b));
                    }

                    // Now you can safely send encodedImage over the data channel
                    dc->send(encodedImage);

                    // std::this_thread::sleep_for(5s); // Adjust the frequency as needed
                }
            }).detach();
		});

		dc->onClosed([id]() { std::cout << "DataChannel from " << id << " closed" << std::endl; });

		dc->onMessage([id](auto data) {
			// data holds either std::string or rtc::binary
			if (std::holds_alternative<std::string>(data))
				std::cout << "Message from " << id << " received: " << std::get<std::string>(data)
				          << std::endl;
			else
				std::cout << "Binary message from " << id
				          << " received, size=" << std::get<rtc::binary>(data).size() << std::endl;
		});

		dataChannelMap.emplace(id, dc);
	});

	peerConnectionMap.emplace(id, pc);
	return pc;
};

// Helper function to generate a random ID
std::string randomId(size_t length) {
	using std::chrono::high_resolution_clock;
	static thread_local std::mt19937 rng(
	    static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));
	static const std::string characters(
	    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
	std::string id(length, '0');
	std::uniform_int_distribution<int> uniform(0, int(characters.size() - 1));
	std::generate(id.begin(), id.end(), [&]() { return characters.at(uniform(rng)); });
	return id;
}
