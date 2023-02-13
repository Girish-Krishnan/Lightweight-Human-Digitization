#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <filesystem>
#include <numpy/npy_io.h>

std::mutex mtx;

void capture_frame(rs2::pipeline& pipe, int cam_index, const std::string& cam_serial)
{
    
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::align align(RS2_STREAM_COLOR);
        frames = align.process(frames);
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        mtx.lock();
        std::cout << "Captured aligned image and depth map from camera " << cam_index << std::endl;

        // Convert the RealSense frames to OpenCV Mat format
        cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_image(cv::Size(depth_frame.get_width(), depth_frame.get_height()), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        // Create a directory for the camera
        // std::filesystem::create_directory(cam_serial);

        // Save the images to disk
        std::string color_filename = "./" + cam_serial + "/" + "image_cpp_only.jpg";
        std::string depth_image_filename = "./" + cam_serial + "/" + "depth_cpp_only.png";
        std::string depth_filename = cam_serial + "/" + "depth_map_cpp_only.npy";

        NumpyFile np_file(depth_filename.c_str(), "w");
        np_file.write((char*)depth_image.data, depth_image.total() * depth_image.elemSize());
        cv::imwrite(color_filename, color_image);
        cv::imwrite(depth_image_filename, depth_image);
        mtx.unlock();
    
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

    // Start a separate thread for each camera to capture frames
    std::vector<std::thread> threads;
    for (int i = 0; i < cam_count; i++)
    {
        rs2::device dev = pipes[i].get_active_profile().get_device();
        std::string cam_serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    threads.emplace_back(capture_frame, std::ref(pipes[i]), i, cam_serial);
    }

    // Wait for all threads to finish
for (int i = 0; i < cam_count; i++)
{
    threads[i].join();
}

return 0;

}