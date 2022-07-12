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
bool MotionEstimator::verifySolutions(
    const std::vector<Eigen::Matrix3f>& R10_vec, const std::vector<Eigen::Vector3f>& t10_vec, 
    const PixelVec& pxvec0, const PixelVec& pxvec1, const std::shared_ptr<Camera>& cam,
    Eigen::Matrix3f& R10_true, Eigen::Vector3f& t10_true, 
    MaskVec& max_inlier, PointVec& X1_true)
{
    bool success = true;

    if(pxvec0.size() != pxvec1.size() )
        throw std::runtime_error("pxvec0.size() != pxvec1.size()");

    int n_pts = pxvec0.size(); 

	// Extract homogeneous 2D point which is inliered with essential constraint
	// Find reasonable rotation and translational vector
	int max_num = 0;
	for( uint32_t i = 0; i < R10_vec.size(); i++ ){
		Eigen::Matrix3f R10 = R10_vec[i];
		Eigen::Vector3f t10 = t10_vec[i];
		
		PointVec X0, X1;
		MaskVec maskvec_inlier(n_pts, false);
        triangulateDLT(pxvec0, pxvec1, 
                        R10, t10, cam,
                        X0, X1);

        
		int num_inlier = 0;
		for(int i = 0; i < n_pts; ++i){
            if(X0[i](2) > 0 && X1[i](2) > 0){
                ++num_inlier;
                maskvec_inlier[i] = true;
            }
        }

		if( num_inlier > max_num ){
			max_num = num_inlier;
			max_inlier = maskvec_inlier;
            X1_true    = X1;
			R10_true   = R10;
			t10_true   = t10;
		}
	}

	// Store 3D position in features
	return success;
};

void MotionEstimator::triangulateDLT(const PixelVec& pts0, const PixelVec& pts1, 
                    const Eigen::Matrix3f& R10, const Eigen::Vector3f& t10, const std::shared_ptr<Camera>& cam, 
                    PointVec& X0, PointVec& X1)
{

    if(pts0.size() != pts1.size() )
        throw std::runtime_error("pts0.size() != pts1.size()");

    int n_pts = pts0.size(); 

    float fxinv = cam->fxinv();
    float fyinv = cam->fyinv();
    float cx = cam->cx();
    float cy = cam->cy();

    Eigen::MatrixXf& M = m_matrix_template_;
    M.resize(3*n_pts, n_pts + 1);
    M.setZero();

    PointVec x0, x1;
    x0.resize(n_pts); x1.resize(n_pts);
    for(int i = 0; i < n_pts; ++i){
        const Pixel& p0 = pts0[i];
        const Pixel& p1 = pts1[i];

        x0[i] << (p0.x-cx)*fxinv, (p0.y-cy)*fyinv, 1.0;
        x1[i] << (p1.x-cx)*fxinv, (p1.y-cy)*fyinv, 1.0;

        Eigen::Matrix3f skewx1 = skew(x1[i]);
        
        M.block(3*i,i,3,1)      = skewx1*R10*x0[i];
        M.block(3*i,n_pts, 3,1) = skewx1*t10;
    }

    // Solve SVD
    Eigen::MatrixXf V;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeFullV);
    V = svd.matrixV();

    X0.resize(n_pts);
    X1.resize(n_pts);
    float normalizer = 1.0f/V(n_pts, n_pts);
    for(int i = 0; i < n_pts; ++i){
        X0[i] = V(i,n_pts)*normalizer*x0[i];
        X1[i] = R10*X0[i]+t10;
    }
};

Eigen::Matrix3f MotionEstimator::skew(const Eigen::Vector3f& vec){
    Eigen::Matrix3f mat;
    mat << 0.0, -vec(2), vec(1), 
          vec(2), 0.0, -vec(0),
          -vec(1),vec(0),0.0;
    return mat;
};