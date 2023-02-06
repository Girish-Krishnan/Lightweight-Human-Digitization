#include <librealsense2/rs.hpp>
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void capture_frame(rs2::pipeline& pipe, int cam_index)
{
    while (true)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        mtx.lock();
        std::cout << "Captured image and depth map from camera " << cam_index << std::endl;
        // Perform any desired processing on the color_frame and depth_frame here
        mtx.unlock();
    }
}

int main()
{
    rs2::context ctx;
    std::vector<rs2::pipeline> pipes;

    // Query all currently connected RealSense D415 cameras
    for (auto&& dev : ctx.query_devices())
    {
        if (dev.get_info(RS2_CAMERA_INFO_NAME) == "Intel RealSense D415")
        {
            rs2::pipeline pipe(ctx);
            pipe.start(dev);
            pipes.push_back(pipe);
        }
    }

    int cam_count = pipes.size();
    std::cout << "Found " << cam_count << " Intel RealSense D415 cameras" << std::endl;

    // Start a separate thread for each camera to capture frames synchronously
    std::vector<std::thread> threads;
    for (int i = 0; i < cam_count; i++)
    {
        threads.push_back(std::thread(capture_frame, std::ref(pipes[i]), i));
    }

    // Wait for all threads to finish
    for (auto& t : threads)
    {
        t.join();
    }

    return 0;
}
