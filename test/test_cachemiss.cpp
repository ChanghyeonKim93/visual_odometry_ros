#include <iostream>
#include <vector>

#include <string>
#include <sstream>
#include <exception>

#include <random>

#include <opencv2/core.hpp>
#include <ros/ros.h>

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
        int n_cols = 1920;
        int n_rows = 1280;
        cv::Mat img = cv::Mat::zeros(cv::Size(n_cols, n_rows), CV_8UC1);

        unsigned char *ptr = img.data;
        unsigned char *ptr_end = ptr + img.step * img.rows;
        for (; ptr < ptr_end; ++ptr)
        {
            *ptr = (unsigned char)dist(gen);
        }

        // TEST 1. summation row-major & col-major
        int max_iter = 1000;
        // Cache hit case!
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
                    sum += *ptr;
            }
            res1.push_back(sum); // This line is necessary for dynamic situation.
        }
        timer::toc(1);

        // Cache miss case!
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
                    sum += *ptr;
            }
            res2.push_back(sum); // This line is necessary for dynamic situation.
        }
        timer::toc(1);

        // Verify results
        for (int iter = 0; iter < max_iter; ++iter)
            if (res1[iter] != res2[iter])
                std::runtime_error("Different result occurs!");

        // TEST 2. branch prediction
        int n_data = 1e6;
        std::vector<int> values(n_data, 0);
        for (auto &it : values)
            it = dist(gen); /* value range = [0,255] */

        int threshold = 127;

        // Branch prediction: fail
        timer::tic();
        res1.resize(0);
        for (int iter = 0; iter < max_iter; ++iter)
        {
            int sum = 0;
            for (const auto &it : values)
            {
                if (it > threshold)
                    sum += it;
            }
            res1.push_back(sum); // This line is necessary for dynamic situation.
        }
        timer::toc(1);

        // Branch prediction: success
        std::sort(values.begin(), values.end());
        timer::tic();
        res2.resize(0);
        for (int iter = 0; iter < max_iter; ++iter)
        {
            int sum = 0;
            for (const auto &it : values)
            {
                if (it > threshold)
                    sum += it;
            }
            res2.push_back(sum); // This line is necessary for dynamic situation.
        }
        timer::toc(1);
    }
    catch (std::exception &e)
    {
        ROS_ERROR(e.what());
    }

    ROS_INFO_STREAM("test_cachemiss - TERMINATED.");
    return -1;
}
