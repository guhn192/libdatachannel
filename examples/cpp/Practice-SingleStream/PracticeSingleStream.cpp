#include "window.hpp"

#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/Error.hpp"
#include <mutex>
#include <thread>

int main(int argc, char **argv) try {
    
    // create frame resource
    std::mutex                 videoFrameMutex;
    std::shared_ptr<ob::Frame> colorFrame;
    std::shared_ptr<ob::Frame> depthFrame;

    std::vector<std::shared_ptr<ob::Frame>> frames;

    // Create a pipeline with default device
    ob::Pipeline pipe;

    // Configure which streams to enable or disable for the Pipeline by creating a Config
    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

    try {
        auto colorProfiles = pipe.getStreamProfileList(OB_SENSOR_COLOR);
        auto colorProfile  = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
        config->enableStream(colorProfile->as<ob::VideoStreamProfile>());
    }
    catch(...) {
        std::cout << "color stream not found!" << std::endl;
    }
    try {
        auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
        auto depthProfile  = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
        config->enableStream(depthProfile->as<ob::VideoStreamProfile>());
    }
    catch(...) {
        std::cout << "depth stream not found!" << std::endl;
    }

    // Configure the alignment mode as hardware D2C alignment
    config->setAlignMode(ALIGN_D2C_HW_MODE);

    // Start the pipeline with config

    pipe.start(config, [&](std::shared_ptr<ob::FrameSet> frameset) {
        std::unique_lock<std::mutex> lk(videoFrameMutex);
        colorFrame = frameset->colorFrame();
        depthFrame = frameset->depthFrame();
    });

    auto                        dev = pipe.getDevice();

    // Create a window for rendering and set the resolution of the window
    Window app("MultiStream", 1280, 720, RENDER_GRID);
    frames.resize(5);

    while(app) {
        {
            std::unique_lock<std::mutex> lock(videoFrameMutex);
            if(colorFrame) {
                frames.at(0) = colorFrame;
            }
            if(depthFrame) {
                frames.at(1) = depthFrame;
            }
        }
        std::vector<std::shared_ptr<ob::Frame>> framesForRender;
        for(auto &frame: frames) {
            if(frame) {
                framesForRender.push_back(frame);
            }
        }
        app.addToRender(framesForRender);
    }

    // Stop the Pipeline, no frame data will be generated
    pipe.stop();
    return 0;
}
catch(ob::Error &e) {
    std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs() << "\mmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
    exit(EXIT_FAILURE);
}
