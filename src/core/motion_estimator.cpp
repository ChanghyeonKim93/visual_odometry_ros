#include "core/motion_estimator.h"

MotionEstimator::MotionEstimator(){

};

MotionEstimator::~MotionEstimator(){
    
};

bool MotionEstimator::calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    Rot3& R10_true, Pos3& t10_true, PointVec& X0_true, MaskVec& mask_inlier)
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
    essential = cv::findEssentialMat(pts0, pts1, cam->cvK(), cv::RANSAC, 0.999, 1.0, inlier_mat);
    // essential = cv::findEssentialMat(pts0, pts1, cam->cvK(), cv::LMEDS, 0.999, 1.0, inlier_mat);
    
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
    std::vector<Rot3> R10_vec(4);
    std::vector<Pos3> t10_vec(4);
    R10_vec[0] = U * W * V.transpose();
    R10_vec[1] = R10_vec[0];
    R10_vec[2] = U * W.transpose() * V.transpose();
    R10_vec[3] = R10_vec[2];

    t10_vec[0] = U.block<3,1>(0, 2);
    t10_vec[1] = -t10_vec[0];
    t10_vec[2] = t10_vec[0];
    t10_vec[3] = -t10_vec[0];

    // Solve two-fold ambiguity
    bool success = true;

    MaskVec  maskvec_verify(n_pts,true);

    Rot3 R10_verify;
    Pos3 t10_verify;

    success = findCorrectRT(R10_vec, t10_vec, pts0, pts1, cam, 
                            R10_verify, t10_verify, maskvec_verify, X0_true);

    R10_true = R10_verify;
    t10_true = t10_verify;
    
    uint32_t cnt_correctRT = 0;
    for( int i = 0; i < n_pts; ++i) {
        mask_inlier[i] = (maskvec_verify[i] && maskvec_5p[i]);
        if(mask_inlier[i]) ++cnt_correctRT; 
    }

    // std::cout << "    5p: " << cnt_5p <<", correctRT: " << cnt_correctRT <<std::endl;
    
    return success;
};

/**
 * @brief PnP 알고리즘
 * @details 3차원 좌표가 생성된 특징점들에 대해 PnP 알고리즘을 수행하여 자세값을 계산한다.
 * @param R 회전 행렬. 초기화하여 입력하여야 한다. 만약, 초기값이 없을 경우 Identity로 넣어야 한다.
 * @param t 변위 벡터. 초기화하여 입력하여야 한다. 만약, 초기값이 없을 경우 setZero() 하여 넣어야 한다.
 * @param maskvec_inlier PnP 인라이어 mask
 * @return 성공적으로 자세를 계산하면, true
 * @author Changhyeon Kim (hyun91015@gmail.com)
 * @date 12-July-2022
 */
bool MotionEstimator::calcPosePnPAlgorithm(const PointVec& Xw, const PixelVec& pts_c, const std::shared_ptr<Camera>& cam, 
    Rot3& Rwc, Pos3& twc, MaskVec& maskvec_inlier){
    if(Xw.size() != pts_c.size())
        throw std::runtime_error("Error in 'calcPosePnPAlgorithm()': Xw.size() != pts_c.size()");

    if(Xw.size() == 0)
        throw std::runtime_error("Error in 'calcPosePnPAlgorithm()': Xw.size() == pts_c.size() == 0");

    int n_pts = Xw.size();
    maskvec_inlier.resize(n_pts, false);

    // cv::SOLVEPNP_AP3P;
    // cv::SOLVEPNP_EPNP;

    std::vector<cv::Point3f> object_pts(n_pts);
    for(int i = 0; i < n_pts; ++i) {
        object_pts[i].x = Xw[i](0);
        object_pts[i].y = Xw[i](1);
        object_pts[i].z = Xw[i](2);
    }

    // Prior values...
    cv::Mat R_cv, r_vec, t_vec;
    cv::eigen2cv(Rwc,   R_cv);
    cv::eigen2cv(twc,   t_vec);
    cv::Rodrigues(R_cv, r_vec);

    std::vector<int> idx_inlier;
    float pnp_reprojection_error = 0.5; // pixels
    bool flag = cv::solvePnPRansac(object_pts, pts_c, cam->cvK(), cv::noArray(),
                                    r_vec, t_vec, true, 1e3,
                                    pnp_reprojection_error,     0.99, idx_inlier, cv::SOLVEPNP_AP3P);
    if (!flag){
        flag = cv::solvePnPRansac(object_pts, pts_c, cam->cvK(), cv::noArray(),
                                    r_vec, t_vec, true, 1e3,
                                    2.0*pnp_reprojection_error, 0.99, idx_inlier, cv::SOLVEPNP_AP3P);
    }
    
    // change format
    cv::Rodrigues(r_vec, R_cv);
    cv::cv2eigen(R_cv, Rwc);
    cv::cv2eigen(t_vec, twc);

    // Set inliers
    int num_inliers = 0;
    for(int i = 0; i < idx_inlier.size(); ++i){
        maskvec_inlier[idx_inlier[i]] = true;
        ++num_inliers;
    }

    std::cout << "PnP inliers: " << num_inliers << " / " << n_pts << std::endl;

    return flag;
};

bool MotionEstimator::findCorrectRT(
    const std::vector<Rot3>& R10_vec, const std::vector<Pos3>& t10_vec, 
    const PixelVec& pxvec0, const PixelVec& pxvec1, const std::shared_ptr<Camera>& cam,
    Rot3& R10_true, Pos3& t10_true, 
    MaskVec& maskvec_true, PointVec& X0_true)
{
    bool success = true;

    if(pxvec0.size() != pxvec1.size() )
        throw std::runtime_error("Error in 'findCorrectRT()': pxvec0.size() != pxvec1.size()");

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
        Mapping::triangulateDLT(pxvec0, pxvec1, R10, t10, cam,
                                X0, X1);

        // Check chirality
		int num_inlier = 0;
		for(int i = 0; i < n_pts; ++i) {
            if(X0[i](2) > 0 && X1[i](2) > 0) {
                ++num_inlier;
                maskvec_inlier[i] = true;
            }
        }

        // Maximum inlier?
		if( num_inlier > max_num_inlier ){
			max_num_inlier = num_inlier;
			maskvec_true   = maskvec_inlier;
            X0_true        = X0;
			R10_true       = R10;
			t10_true       = t10;
		}
	}

    // 만약, 전체 점 갯수 대비 max_num_inlier가 60 % 이상이 아니면 실패
    if(max_num_inlier < 0.5 * (double)n_pts) {
        std::cout << "WARRNING - too low number of inliers!\n";
        // success = false;
    }
        
	return success;
};

float MotionEstimator::findInliers1PointHistogram(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    MaskVec& maskvec_inlier){
    
    if(pts0.size() != pts1.size()) {
        throw std::runtime_error("Error in 'fineInliers1PointHistogram()': pts0.size() != pts1.size()");
        return false;
    }

    int n_pts = pts0.size();

    maskvec_inlier.resize(n_pts,false);

    float invfx = cam->fxinv();
    float invfy = cam->fyinv();
    float cx = cam->cx();
    float cy = cam->cy();

    std::vector<float> theta(n_pts);
    for(int i = 0; i < n_pts; ++i){
        float x0 = (pts0[i].x-cx)*invfx;
        float y0 = (pts0[i].y-cy)*invfy;
        float x1 = (pts1[i].x-cx)*invfx;
        float y1 = (pts1[i].y-cy)*invfy;
        float z0 = 1; float z1 = 1;

        float val = (x0*y1-y0*x1) / (y0*z1+z0*y1);
        float th  = -2.0*atan(val);
        theta[i] = th;
    }

    // Make theta histogram vector.
    float hist_min = -0.5; // radian
    float hist_max =  0.5; // radian
    float n_bins   =  400;
    std::vector<float> hist_centers;
    std::vector<int>   hist_counts;
    histogram::makeHistogram<float>(theta, hist_min, hist_max, n_bins, hist_centers, hist_counts);
    float th_opt = histogram::medianHistogram(hist_centers, hist_counts);

    // std::cout << "theta_optimal: " << th_opt << " rad\n";

    Rot3 R10;
    Pos3 t10;
    float costh = cos(th_opt);
    float sinth = sin(th_opt);
    R10 << costh, 0, sinth, 0, 1, 0, -sinth, 0, costh;
    t10 << sin(th_opt*0.5f), 0.0f, cos(th_opt*0.5f);

    std::vector<float> sampson_dist;
    this->calcSampsonDistance(pts0, pts1, cam, R10, t10, sampson_dist);

    float thres_sampson = 10.0; // 15.0 px
    thres_sampson *= thres_sampson;
    for(int i = 0; i < n_pts; ++i){
        if(sampson_dist[i] <= thres_sampson) maskvec_inlier[i] = true;
        else maskvec_inlier[i] = false;
        // std::cout << i << " -th sampson dist: " << sampson_dist[i] << " px\n";
    }


    return th_opt;
};


void MotionEstimator::calcSampsonDistance(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam, 
    const Rot3& R10, const Pos3& t10, std::vector<float>& sampson_dist)
{
    if(pts0.size() != pts1.size()) 
        throw std::runtime_error("Error in 'fineInliers1PointHistogram()': pts0.size() != pts1.size()");
    
    int n_pts = pts0.size();
    
    sampson_dist.resize(n_pts);

    Eigen::Matrix3f E10,F10, F10t;

    E10 = Mapping::skew(t10)*R10;
    F10 = cam->Kinv().transpose()*E10*cam->Kinv();
    F10t = F10.transpose();

    for(int i = 0; i < n_pts; ++i){
        Point p0,p1;
        p0 << pts0[i].x, pts0[i].y, 1.0f;
        p1 << pts1[i].x, pts1[i].y, 1.0f;

        Point F10p0  = F10*p0;
        Point F10tp1 = F10t*p1;
        
        float numerator = p1.transpose()*F10p0;
        numerator *= numerator;
        float denominator = F10p0(0)*F10p0(0) + F10p0(1)*F10p0(1) + F10tp1(0)*F10tp1(0) + F10tp1(1)*F10tp1(1);
        float dist_tmp = numerator / denominator;
        sampson_dist[i] = dist_tmp;
    }
};