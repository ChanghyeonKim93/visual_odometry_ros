#include "core/motion_estimator.h"

MotionEstimator::MotionEstimator(){

};

MotionEstimator::~MotionEstimator(){
    
};

bool MotionEstimator::calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    MaskVec& mask_inlier)
{
    std::cout <<" - MotionEstimator - 'calcPose5PointsAlgorithm()'\n";
    if(pts0.size() != pts1.size()) {
        throw std::runtime_error("calcPose5PointsAlgorithm(): pts0.size() != pts1.size()");
        return false;
    }
    if(pts0.size() == 0) {
        throw std::runtime_error("calcPose5PointsAlgorithm(): pts0.size() == pts1.size() == 0");
        return false;
    }
    int n_pts = pts0.size();
    mask_inlier.resize(n_pts, true);

    // Calculate essential matrix
    cv::Mat inlier_mat, essential;
    essential = cv::findEssentialMat(pts0, pts1, cam->cvK(), cv::RANSAC, 0.999, 1.5, inlier_mat);
    
    // Calculate fundamental matrix
    Eigen::Matrix3f E10, F10;
    cv::cv2eigen(essential, E10);
    F10 = cam->Kinv().transpose() * E10 * cam->Kinv();

    // Check inliers
    uint32_t cnt_inlier = 0;
    bool* ptr_inlier = inlier_mat.ptr<bool>(0);
    std::vector<uint32_t> idx_inlier;
    for(int i = 0; i < inlier_mat.rows; ++i){
        if (ptr_inlier[i]){
            idx_inlier.push_back(i);
            ++cnt_inlier;
        } 
        else mask_inlier[i] = false;
    }

    // Extract R, t
    Eigen::Matrix3f U,V;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E10, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();

    if (U.determinant() < 0)
        U.block(0, 2, 3, 1) = -U.block(0, 2, 3, 1);
    if (V.determinant() < 0)
        V.block(0, 2, 3, 1) = -V.block(0, 2, 3, 1);

    Eigen::Matrix3f W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    // Four possibilities.
    std::vector<Eigen::Matrix3f> R10_vec(4);
    std::vector<Eigen::Vector3f> t10_vec(4);
    R10_vec[0] = U * W * V.transpose();
    R10_vec[1] = R10_vec[0];
    R10_vec[2] = U * W.transpose() * V.transpose();
    R10_vec[3] = R10_vec[2];
    t10_vec[0] = U.block(0, 2, 3, 1);
    t10_vec[1] = -t10_vec[0];
    t10_vec[2] = t10_vec[0];
    t10_vec[3] = -t10_vec[0];

    // Solve two-fold ambiguity
    std::vector<bool> max_inlier;
    std::vector<Eigen::Vector3f> X_curr;

    bool success = true;

    return success;

    // if( !verifySolutions(R_vec, t_vec, R_, t_, max_inlier, X_curr) ) return false;
    // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "## Verify unique pose: " << lsi::toc() << std::endl;

    // for( int i = 0; i < max_inlier.size(); i++ ){
    //     if( max_inlier[i] ){
    //         features_[idx_2d_inlier[i]].point_curr = (Eigen::Vector4d() << X_curr[i], 1).finished();
    //         features_[idx_2d_inlier[i]].is_3D_reconstructed = true;
    //         num_feature_3D_reconstructed_++;
    //     }
    // }

    // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "# Triangulation: " << lsi::toc() << std::endl;
    // if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "! Extract essential between: " << curr_keyframe_.id << " <--> " << curr_frame_.id << std::endl;

    // if (num_feature_2D_inliered_ < params_.th_inlier){
    //     if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There are a few inliers matching features in 2D" << std::endl;
    //     return false;
    // }else{
    //     is_start_ = true;
    //     return true;
    // }
};

bool MotionEstimator::calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts1){

};
bool MotionEstimator::verifySolutions(const std::vector<Eigen::Matrix3f>& R10_vec,
                        const std::vector<Eigen::Vector3f>& t10_vec, 
                        Eigen::Matrix3f& R10_true, 
                        Eigen::Vector3f& t10_true, 
                        MaskVec& max_inlier, 
                        PointVec& X0)
{
    bool success;

	// // Extract homogeneous 2D point which is inliered with essential constraint
	// std::vector<int> idx_2D_inlier;
	// for( int i = 0; i < num_feature_; i++ )
	// 	if( features_[i].is_2D_inliered )
	// 		idx_2D_inlier.emplace_back(i);

    // int key_idx;
	// std::vector<cv::Point2f> uv_prev(idx_2D_inlier.size()), uv_curr(idx_2D_inlier.size());
    // for (uint32_t i = 0; i < idx_2D_inlier.size(); i++){
    //     key_idx = features_[idx_2D_inlier[i]].life - 1 - (step_ - keystep_);
    //     uv_prev[i] = features_[idx_2D_inlier[i]].uv[key_idx];
    //     uv_curr[i] = features_[idx_2D_inlier[i]].uv.back();
    // }

	// // Find reasonable rotation and translational vector
	// int max_num = 0;
	// for( uint32_t i = 0; i < R_vec.size(); i++ ){
	// 	Eigen::Matrix3d R1 = R_vec[i];
	// 	Eigen::Vector3d t1 = t_vec[i];
		
	// 	std::vector<Eigen::Vector3d> X_prev, X_curr;
	// 	std::vector<bool> inlier;
	// 	constructDepth(uv_prev, uv_curr, R1, t1, X_prev, X_curr, inlier);

	// 	int num_inlier = 0;
	// 	for( uint32_t i = 0; i < inlier.size(); i++ )
	// 		if( inlier[i] )
	// 			num_inlier++;

	// 	if( num_inlier > max_num ){
	// 		max_num = num_inlier;
	// 		max_inlier = inlier;
    //         opt_X_curr = X_curr;
			
	// 		R = R1;
	// 		t = t1;
	// 	}
	// }

	// // Store 3D position in features
	// if( max_num < num_feature_2D_inliered_*0.5 ){
    //     if( MVO::s_file_logger_.is_open() ) MVO::s_file_logger_ << "Warning: There is no verified solution" << std::endl;
	// 	success = false;
	// }else{
	// 	success = true;
	// }

	return success;
};