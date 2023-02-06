#include <boost/python.hpp>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

void capture_color_depth_images(const std::string& color_filename, const std::string& depth_filename)
{
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    rs2::frameset frames = pipe.wait_for_frames();
    rs2::frame color_frame = frames.get_color_frame();
    rs2::frame depth_frame = frames.get_depth_frame();

    cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat depth_image(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

    cv::imwrite(color_filename, color_image);
    cv::imwrite(depth_filename, depth_image);
}

BOOST_PYTHON_MODULE(capture)
{
    using namespace boost::python;
    def("capture_color_depth_images", capture_color_depth_images);
}
