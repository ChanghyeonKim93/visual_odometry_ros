#include <iostream>
#include <vector>

#include <string>
#include <sstream>
#include <exception>

#include <random>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "core/util/timer.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_cachemiss");
    ros::NodeHandle nh("~");

    ROS_INFO_STREAM("test_cachemiss - starts.");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    try
    {
        int n_cols = 1280;
        int n_rows = 960;
        cv::Mat img = cv::Mat::zeros(cv::Size(n_cols, n_rows), CV_8UC1);

        unsigned char *ptr = img.data;
        unsigned char *ptr_end = ptr + img.step * img.rows;
        for (; ptr < ptr_end; ++ptr)
        {
            *ptr = (unsigned char)dist(gen);
        }

        // Summation
        int max_iter = 10000;
        // Cache hit!
        timer::tic();
        std::vector<int> res1;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            int sum = 0;
            for (int v = 0; v < img.rows; ++v)
            {
                unsigned char *ptr = img.data + v * img.cols;
                const unsigned char *ptr_end = ptr + img.cols;
                for (; ptr < ptr_end; ++ptr)
                {
                    sum += *ptr;
                }
            }
            res1.push_back(sum);
        }
        timer::toc(1);

        // Cache miss!
        timer::tic();
        std::vector<int> res2;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            int sum = 0;
            for (int u = 0; u < img.cols; ++u)
            {
                unsigned char *ptr = img.data + u;
                const unsigned char *ptr_end = ptr + u + (img.rows - 1) * img.cols;
                for (; ptr <= ptr_end; ptr += img.cols)
                {
                    sum += *ptr;
                }
            }
            res2.push_back(sum);
        }
        timer::toc(1);

        // Verify results
        for (int iter = 0; iter < max_iter; ++iter)
            if (res1[iter] != res2[iter])
                std::runtime_error("Different result occurs!");
    }
    catch (std::exception &e)
    {
        ROS_ERROR(e.what());
    }

    ROS_INFO_STREAM("test_cachemiss - TERMINATED.");
    return -1;
}
