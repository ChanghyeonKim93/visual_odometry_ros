#include "core/motion_estimator.h"

MotionEstimator::MotionEstimator(){

};

MotionEstimator::~MotionEstimator(){
    
};

bool MotionEstimator::calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    Eigen::Matrix3f& R10_true, Eigen::Vector3f& t10_true, MaskVec& mask_inlier)
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
    uint32_t cnt_5p = 0;
    bool* ptr_inlier = inlier_mat.ptr<bool>(0);
    MaskVec maskvec_5p(n_pts, false);
    for(int i = 0; i < inlier_mat.rows; ++i){
        if (ptr_inlier[i]){
            maskvec_5p[i] = true;
            ++cnt_5p;
        } 
        else maskvec_5p[i] = false;
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
    bool success = true;

    MaskVec  maskvec_verify(n_pts,true);
    PointVec X1_verify;

    Eigen::Matrix3f R10_verify;
    Eigen::Vector3f t10_verify;

    success = findCorrectRT(R10_vec, t10_vec, pts0, pts1, cam, 
                            R10_verify, t10_verify, maskvec_verify, X1_verify);
    
    std::cout <<"R:\n" << R10_verify << "\n";
    std::cout <<"t:\n" << t10_verify << "\n";

    uint32_t cnt_correctRT = 0;
    for( int i = 0; i < n_pts; ++i) {
        mask_inlier[i] = (maskvec_verify[i] && maskvec_5p[i]);
        if(mask_inlier[i]) ++cnt_correctRT; 
    }

    std::cout << "    5p: " << cnt_5p <<", correctRT: " << cnt_correctRT <<std::endl;
    
    return success;
};

bool MotionEstimator::calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts1){

};
bool MotionEstimator::findCorrectRT(
    const std::vector<Eigen::Matrix3f>& R10_vec, const std::vector<Eigen::Vector3f>& t10_vec, 
    const PixelVec& pxvec0, const PixelVec& pxvec1, const std::shared_ptr<Camera>& cam,
    Eigen::Matrix3f& R10_true, Eigen::Vector3f& t10_true, 
    MaskVec& maskvec_true, PointVec& X1_true)
{
    bool success = true;

    if(pxvec0.size() != pxvec1.size() )
        throw std::runtime_error("pxvec0.size() != pxvec1.size()");

    int n_pts = pxvec0.size(); 

	// Extract homogeneous 2D point which is inliered with essential constraint
	// Find reasonable rotation and translational vector
	int max_num_inlier = 0;
	for( uint32_t i = 0; i < R10_vec.size(); i++ ){
        // i-th pose candidate
		const Eigen::Matrix3f& R10 = R10_vec[i];
		const Eigen::Vector3f& t10 = t10_vec[i];
		
        // Triangulate by the i-th pose candidate
		PointVec X0, X1;
		MaskVec maskvec_inlier(n_pts, false);
        Mapping::triangulateDLT(pxvec0, pxvec1, 
                        R10, t10, cam,
                        X0, X1);

        // Check chirality
		int num_inlier = 0;
		for(int i = 0; i < n_pts; ++i){
            if(X0[i](2) > 0 && X1[i](2) > 0){
                ++num_inlier;
                maskvec_inlier[i] = true;
            }
        }

        // Maximum inlier?
		if( num_inlier > max_num_inlier ){
			max_num_inlier = num_inlier;
			maskvec_true   = maskvec_inlier;
            X1_true        = X1;
			R10_true       = R10;
			t10_true       = t10;
		}
	}

    // 만약, 전체 점 갯수 대비 max_num_inlier가 60 % 이상이 아니면 실패
    if(max_num_inlier < 0.6 * (double)n_pts) success = false;
        
	return success;
};