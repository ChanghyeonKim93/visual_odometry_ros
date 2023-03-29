#include <iostream>
#include <vector>

#include <time.h>
#include <string>
#include <sstream>
#include <exception>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "core/util/signal_handler_linux.h"

// https://roomedia.tistory.com/entry/60일차-C-openCV-4-실시간-영상에서-ORB-특징점-추출-및-FLANN-매칭하기
//  custom
#include "core/visual_odometry/feature_extractor.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_orb_matching_node");
    ros::NodeHandle nh("~");

    ROS_INFO_STREAM("test_orb_matching_node - starts.");

    try
    {
        std::cout << "RUN!\n";

        // Make ORB extractor
        int n_cols = 1241;
        int n_rows = 376;
        int n_bins_u = 40;
        int n_bins_v = 24;
        int THRES_FAST = 20.0;
        int radius = 0;

        std::shared_ptr<FeatureExtractor> ext_;
        ext_ = std::make_shared<FeatureExtractor>();
        ext_->initParams(n_cols, n_rows, n_bins_u, n_bins_v, THRES_FAST, radius);

        // Load images
        std::vector<cv::Mat> imgs(6);
        imgs.at(0) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/0.png", cv::IMREAD_GRAYSCALE);
        imgs.at(1) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/1.png", cv::IMREAD_GRAYSCALE);
        imgs.at(2) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/2.png", cv::IMREAD_GRAYSCALE);
        imgs.at(3) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/3.png", cv::IMREAD_GRAYSCALE);
        imgs.at(4) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/4.png", cv::IMREAD_GRAYSCALE);
        imgs.at(5) = cv::imread("/home/larrkchlinux/Documents/kitti/test_orbmatching/5.png", cv::IMREAD_GRAYSCALE);

        // Extract feature and draw images
        ext_->resetWeightBin();
        std::vector<cv::KeyPoint> kpts0;
        cv::Mat desc0;
        ext_->extractAndComputeORB(imgs.at(0), kpts0, desc0);

        for (int i = 1; i < 6; ++i)
        {
            ext_->resetWeightBin();
            std::vector<cv::KeyPoint> kpts;
            cv::Mat desc;
            ext_->extractAndComputeORB(imgs.at(i), kpts, desc);
            cv::Mat desc_tmp = desc.row(0);
            std::cout << desc_tmp << std::endl;

            std::cout << "n_pts : " << kpts.size() << ", desc size (r,c): " << desc.rows << "," << desc.cols << std::endl;

            cv::Mat img_draw;
            imgs.at(i).copyTo(img_draw);
            cv::cvtColor(img_draw, img_draw, CV_GRAY2RGB);

            for (int j = 0; j < kpts.size(); ++j)
                cv::circle(img_draw, kpts.at(j).pt, 1.0, cv::Scalar(0, 0, 255), 1); // alived magenta
            std::string window_name = "img" + i;
            cv::namedWindow(window_name);
            cv::imshow(window_name, img_draw);
            cv::waitKey(0);

            // 매칭...
            int TH_HIGH = 100;
            int TH_LOW = 50;

            int nmatches = 0;
            float mfNNratio = 0.6;

            for (int j = 0; j < kpts0.size(); ++j)
            {
                cv::Mat desc_now; // j-th keypoint의 descriptor. 1 행 32열 (32 열 * 1바이트 * 8비트 = 256비트)

                int bestDist = 256;
                int bestLevel = -1;
                int bestDist2 = 256;
                int bestLevel2 = -1;
                int bestIdx = -1;
                // 1) 근처의 keypoints의 INDEX를 찾는다.
                // --> vIndices (vector<int>)

                std::vector<int> vIndices;
                for (int k = 0; k < vIndices.size(); ++k)
                {
                    // 2) 해당 index의 descriptor를 가져온다.
                    const size_t idx = vIndices.at(k);
                    const cv::Mat &d = desc.row(idx);

                    const int dist = ext_->descriptorDistance(desc_now, d);

                    if (dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestLevel2 = bestLevel;
                        // bestLevel = F.mvKeysUn[idx].octave;
                        bestIdx = idx;
                    }
                    else if (dist < bestDist2)
                    {
                        // bestLevel2 = F.mvKeysUn[idx].octave;
                        bestDist2 = dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if (bestDist <= TH_HIGH)
                {
                    if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                        continue;

                    // F.mvpMapPoints[bestIdx]=pMP;
                    nmatches++;
                }
            }
        }
    }
    catch (std::exception &e)
    {
        ROS_ERROR(e.what());
    }

    ROS_INFO_STREAM("test_orb_matching_node - TERMINATED.");
    return -1;
}