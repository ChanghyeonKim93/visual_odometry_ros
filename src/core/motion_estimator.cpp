#include "core/motion_estimator.h"

MotionEstimator::MotionEstimator(){
    thres_1p_ = 10.0; // pixels
    thres_5p_ = 1.5; // pixels
};

MotionEstimator::~MotionEstimator(){
    
};

bool MotionEstimator::calcPose5PointsAlgorithm(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    Rot3& R10_true, Pos3& t10_true, PointVec& X0_true, MaskVec& mask_inlier)
{
    // std::cout <<" - MotionEstimator - 'calcPose5PointsAlgorithm()'\n";
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
    essential = cv::findEssentialMat(pts0, pts1, cam->cvK(), cv::RANSAC, 0.999, thres_5p_, inlier_mat);
    // essential = cv::findEssentialMat(pts0, pts1, cam->cvK(), cv::LMEDS, 0.999, 1.0, inlier_mat);
    
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

    // Calculate fundamental matrix
    Mat33 E10, F10;
    cv::cv2eigen(essential, E10);

    // Refine essential matrix
    // refineEssentialMatIRLS(pts0,pts1, maskvec_5p, cam, E10);
    // refineEssentialMat(pts0,pts1, maskvec_5p, cam, E10);

    F10 = cam->Kinv().transpose() * E10 * cam->Kinv();


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
    float pnp_reprojection_error = 1.5; // pixels
    // bool flag = cv::solvePnPRansac(object_pts, pts_c, cam->cvK(), cv::noArray(),
    //                                 r_vec, t_vec, true, 1e3,
    //                                 pnp_reprojection_error,     0.99, idx_inlier, cv::SOLVEPNP_AP3P);
    // if (!flag){
    //     flag = cv::solvePnPRansac(object_pts, pts_c, cam->cvK(), cv::noArray(),
    //                                 r_vec, t_vec, true, 1e3,
    //                                 2.0*pnp_reprojection_error, 0.99, idx_inlier, cv::SOLVEPNP_AP3P);
    // }
    bool flag = cv::solvePnPRansac(object_pts, pts_c, cam->cvK(), cv::noArray(),
                                    r_vec, t_vec, true, 1e3,
                                    pnp_reprojection_error, 0.99, idx_inlier, cv::SOLVEPNP_EPNP);
    if (!flag){
        flag = cv::solvePnPRansac(object_pts, pts_c, cam->cvK(), cv::noArray(),
                                    r_vec, t_vec, true, 1e3,
                                    2 * pnp_reprojection_error, 0.99, idx_inlier, cv::SOLVEPNP_EPNP);
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

    if((float)num_inliers/(float)idx_inlier.size() < 0.6f) flag = false;

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


void MotionEstimator::refineEssentialMat(const PixelVec& pts0, const PixelVec& pts1, const MaskVec& mask, const std::shared_ptr<Camera>& cam,
    Mat33& E)
{

    if(pts0.size() != pts1.size() )
        throw std::runtime_error("Error in 'refineEssentialMat()': pts0.size() != pts1.size()");

    int n_pts = pts0.size(); 
    int n_pts_valid = 0;
    for(int i = 0; i < n_pts; ++i){
        if(mask[i]) ++n_pts_valid;
    }

    Eigen::MatrixXf M, M_weight;
    M = Eigen::MatrixXf(n_pts_valid,9);
    M_weight = Eigen::MatrixXf(n_pts_valid,9);

    std::cout <<"size: " << M.rows() << ", " <<  M.cols() <<std::endl;

    int idx = 0;
    float fx = cam->fx();
    float fy = cam->fy();
    float fxinv = cam->fxinv();
    float fyinv = cam->fyinv();
    float cx = cam->cx();
    float cy = cam->cy();

    for(int i = 0; i < n_pts; ++i) {
        if(mask[i]){
            float x0 = fxinv*(pts0[i].x-cx);
            float y0 = fyinv*(pts0[i].y-cy);
            float x1 = fxinv*(pts1[i].x-cx);
            float y1 = fyinv*(pts1[i].y-cy);
            M(idx,0) = x0*x1;
            M(idx,1) = y0*x1;
            M(idx,2) = x1;
            M(idx,3) = x0*y1;
            M(idx,4) = y0*y1;
            M(idx,5) = y1;
            M(idx,6) = x0;
            M(idx,7) = y0;
            M(idx,8) = 1.0f;
            ++idx;
        }
    }

    // solve!
    Eigen::MatrixXf U,V;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();

    Eigen::Matrix<float,9,1> e_vec;
    e_vec(0) = V(0,8);
    e_vec(1) = V(1,8);
    e_vec(2) = V(2,8);
    e_vec(3) = V(3,8);
    e_vec(4) = V(4,8);
    e_vec(5) = V(5,8);
    e_vec(6) = V(6,8);
    e_vec(7) = V(7,8);
    e_vec(8) = V(8,8);

    float inv_last = 1.0f/e_vec(8);
    for(int i = 0; i < 9; ++i){
        e_vec(i) *= inv_last;
    }
    Mat33 E_est ;
    E_est << e_vec(0),e_vec(1),e_vec(2),
            e_vec(3),e_vec(4),e_vec(5),
            e_vec(6),e_vec(7),e_vec(8);

    Eigen::Matrix3f U3, V3, S3;
    Eigen::Vector3f s3;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd3(E_est, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U3 = svd3.matrixU();
    s3 = svd3.singularValues();
    V3 = svd3.matrixV();
    S3 << 1,0,0, 0,1,0, 0,0,0;
    E_est = U3*S3*V3.transpose();

    // std::cout << " e_vec:\n" << (E_est /= E_est(2,2)) << std::endl;
    // std::cout << " Essential : \n" << (E /=E(2,2)) << std::endl;

    E = E_est;

};


void MotionEstimator::refineEssentialMatIRLS(const PixelVec& pts0, const PixelVec& pts1, const MaskVec& mask, const std::shared_ptr<Camera>& cam,
    Mat33& E)
{

    if(pts0.size() != pts1.size() )
        throw std::runtime_error("Error in 'refineEssentialMat()': pts0.size() != pts1.size()");

    int n_pts = pts0.size(); 
    int n_pts_valid = 0;
    for(int i = 0; i < n_pts; ++i){
        if(mask[i]) ++n_pts_valid;
    }

    Eigen::MatrixXf M, M_weight;
    M = Eigen::MatrixXf(n_pts_valid,9);
    M_weight = Eigen::MatrixXf(n_pts_valid,9);

    float fx = cam->fx();
    float fy = cam->fy();
    float fxinv = cam->fxinv();
    float fyinv = cam->fyinv();
    float cx = cam->cx();
    float cy = cam->cy();
    const Mat33& K = cam->K();
    const Mat33& Kinv = cam->Kinv();

    // Precalculation
    int idx = 0;
    for(int i = 0; i < n_pts; ++i) {
        if(mask[i]){
            float x0 = fxinv*(pts0[i].x-cx);
            float y0 = fyinv*(pts0[i].y-cy);
            float x1 = fxinv*(pts1[i].x-cx);
            float y1 = fyinv*(pts1[i].y-cy);
            M(idx,0) = x0*x1;
            M(idx,1) = y0*x1;
            M(idx,2) = x1;
            M(idx,3) = x0*y1;
            M(idx,4) = y0*y1;
            M(idx,5) = y1;
            M(idx,6) = x0;
            M(idx,7) = y0;
            M(idx,8) = 1.0f;
            ++idx;
        }
    }

    // Iterations
    int MAX_ITER = 30;
    for(int iter = 0; iter < MAX_ITER; ++iter){
        Mat33 F10 = Kinv.transpose()*E*Kinv;
        idx = 0;
        for(int i = 0; i < n_pts; ++i) {
            if(mask[i]){
                float weight = 1.0f;

                // Calculate Sampson distance
                float sampson_dist = calcSampsonDistance(pts0[i],pts1[i],F10);
                
                // std::cout << sampson_dist << std::endl;
                if(sampson_dist > 0.001) weight = 0.001/sampson_dist;
                
                M_weight(idx,0) = weight*M(idx,0);
                M_weight(idx,1) = weight*M(idx,1);
                M_weight(idx,2) = weight*M(idx,2);
                M_weight(idx,3) = weight*M(idx,3);
                M_weight(idx,4) = weight*M(idx,4);
                M_weight(idx,5) = weight*M(idx,5);
                M_weight(idx,6) = weight*M(idx,6);
                M_weight(idx,7) = weight*M(idx,7);
                M_weight(idx,8) = weight*M(idx,8);
                ++idx;
            }
        }

        // solve!
        Eigen::MatrixXf U,V;
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(M_weight, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        V = svd.matrixV();

        Eigen::Matrix<float,9,1> e_vec;
        e_vec(0) = V(0,8);
        e_vec(1) = V(1,8);
        e_vec(2) = V(2,8);
        e_vec(3) = V(3,8);
        e_vec(4) = V(4,8);
        e_vec(5) = V(5,8);
        e_vec(6) = V(6,8);
        e_vec(7) = V(7,8);
        e_vec(8) = V(8,8);

        // float inv_last = 1.0f/e_vec(8);
        // for(int i = 0; i < 9; ++i){
        //     e_vec(i) *= inv_last;
        // }
        Mat33 E_est ;
        E_est << e_vec(0),e_vec(1),e_vec(2),
                e_vec(3),e_vec(4),e_vec(5),
                e_vec(6),e_vec(7),e_vec(8);

        Eigen::Matrix3f U3, V3, S3;
        Eigen::Vector3f s3;
        Eigen::JacobiSVD<Eigen::Matrix3f> svd3(E_est, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U3 = svd3.matrixU();
        s3 = svd3.singularValues();
        V3 = svd3.matrixV();
        S3 << 1,0,0, 0,1,0, 0,0,0;
        E_est = U3*S3*V3.transpose();

        // std::cout << iter <<": " << e_vec.transpose() << std::endl;
        // std::cout << " e_vec:\n" << (E_est /= E_est(2,2)) << std::endl;
        // std::cout << " Essential : \n" << (E /=E(2,2)) << std::endl;

        E = E_est;
    }


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
    // this->calcSampsonDistance(pts0, pts1, cam, R10, t10, sampson_dist);
    this->calcSymmetricEpipolarDistance(pts0, pts1, cam, R10, t10, sampson_dist);

    float thres_sampson = thres_1p_; // 15.0 px
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


void MotionEstimator::calcSampsonDistance(const PixelVec& pts0, const PixelVec& pts1,
                        const Mat33& F10, std::vector<float>& sampson_dist)
{
    if(pts0.size() != pts1.size()) 
        throw std::runtime_error("Error in 'fineInliers1PointHistogram()': pts0.size() != pts1.size()");
    
    int n_pts = pts0.size();
    
    sampson_dist.resize(n_pts);

    Eigen::Matrix3f F10t;

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

float MotionEstimator::calcSampsonDistance(const Pixel& pt0, const Pixel& pt1,const Mat33& F10)
{
    Eigen::Matrix3f F10t;

    F10t = F10.transpose();
    Point p0,p1;
    p0 << pt0.x, pt0.y, 1.0f;
    p1 << pt1.x, pt1.y, 1.0f;

    Point F10p0  = F10*p0;
    Point F10tp1 = F10t*p1;
    
    float numerator = p1.transpose()*F10p0;
    numerator *= numerator;
    float denominator = F10p0(0)*F10p0(0) + F10p0(1)*F10p0(1) + F10tp1(0)*F10tp1(0) + F10tp1(1)*F10tp1(1);
    float dist_tmp = numerator / denominator;
    return dist_tmp;
};

void MotionEstimator::calcSymmetricEpipolarDistance(const PixelVec& pts0, const PixelVec& pts1, const std::shared_ptr<Camera>& cam, 
    const Rot3& R10, const Pos3& t10, std::vector<float>& sym_epi_dist)
{
    if(pts0.size() != pts1.size()) 
        throw std::runtime_error("Error in 'fineInliers1PointHistogram()': pts0.size() != pts1.size()");
    
    int n_pts = pts0.size();
    
    sym_epi_dist.resize(n_pts);

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
        float denominator = 1.0f/(F10p0(0)*F10p0(0) + F10p0(1)*F10p0(1)) + 1.0f*(F10tp1(0)*F10tp1(0) + F10tp1(1)*F10tp1(1));
        float dist_tmp = numerator * denominator;
        sym_epi_dist[i] = dist_tmp;
    }
};


void MotionEstimator::setThres1p(float thres_1p){
    thres_1p_ = thres_1p; // pixels
};
void MotionEstimator::setThres5p(float thres_5p){
    thres_5p_ = thres_5p; // pixels
};



bool MotionEstimator::calcPoseOnlyBundleAdjustment(const PointVec& X, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    Rot3& R01_true, Pos3& t01_true, MaskVec& mask_inlier)
{
    // X is represented in the world frame.
    if(X.size() != pts1.size()) 
        throw std::runtime_error("In 'calcPoseOnlyBundleAdjustment()': X.size() != pts1.size().");
    
    bool is_success = true;

    int n_pts = X.size();
    mask_inlier.resize(n_pts);
    
    int MAX_ITER = 250;
    float THRES_HUBER        = 1.0f; // pixels
    float THRES_DELTA_XI     = 1e-7;
    float THRES_DELTA_ERROR  = 1e-5;
    float THRES_REPROJ_ERROR = 5.0f; // pixels

    float lambda = 0.01f;
    float step_size = 1.0f;
    
    float fx = cam->fx(); float fy = cam->fy();
    float cx = cam->cx(); float cy = cam->cy();
    float fxinv = cam->fxinv(); float fyinv = cam->fyinv();

    PoseSE3 T10_init;
    PoseSE3 T01_init;
    T01_init << R01_true, t01_true, 0,0,0,1;
    T10_init = T01_init.inverse();

    PoseSE3Tangent xi10; // se3
    geometry::SE3Log_f(T10_init,xi10);
    
    float err_prev = 1e10f;
    for(uint32_t iter = 0; iter < MAX_ITER; ++iter){
        PoseSE3 T10;
        geometry::se3Exp_f(xi10,T10);

        Rot3 R10 = T10.block<3,3>(0,0);
        Pos3 t10 = T10.block<3,1>(0,3);

        Eigen::Matrix<float,6,6> JtWJ;
        Eigen::Matrix<float,6,1> mJtWr;
        JtWJ.setZero();
        mJtWr.setZero();

        float err_curr = 0.0f;
        float inv_npts = 1.0f/(float)n_pts;
        int cnt_invalid = 0;
        // Warp and project point & calculate error...
        for(int i = 0; i < n_pts; ++i) {
            const Pixel& pt = pts1[i];
            Point Xw = R10*X[i] + t10;

            float iz = 1.0f/Xw(2);
            float xiz = Xw(0)*iz;
            float yiz = Xw(1)*iz;
            float fxxiz = fx*xiz;
            float fyyiz = fy*yiz;

            Pixel pt_warp;
            pt_warp.x = fxxiz+cx;
            pt_warp.y = fyyiz+cy;

            float rx = pt_warp.x - pt.x;
            float ry = pt_warp.y - pt.y;
            
            // Huber weight calculation by the Manhattan distance
            float weight     = 1.0f;
            bool flag_weight = false;

            float absrxry = abs(rx)+abs(ry);
            if(absrxry >= THRES_HUBER){
                weight = THRES_HUBER/absrxry; 
                flag_weight = true;
            } 

            if(absrxry >= THRES_REPROJ_ERROR){
                mask_inlier[i] = false;
                ++cnt_invalid;
            }
            else 
                mask_inlier[i] = true;

            // JtWJ, JtWr for x
            Eigen::Matrix<float,6,1> Jt;
            Jt(0,0) = fx*iz;
            Jt(1,0) = 0.0f;
            Jt(2,0) = -fxxiz*iz;
            Jt(3,0) = -fxxiz*yiz;
            Jt(4,0) = fx*(1.0f+xiz*xiz);
            Jt(5,0) = -fx*yiz;

            if(flag_weight) {
                float w_rx = weight*rx;
                float err = w_rx*rx;
                mJtWr.noalias() -= (w_rx)*Jt;
                JtWJ.noalias() += weight*(Jt*Jt.transpose());
                err_curr += err;
            }
            else {
                float err = rx*rx;
                JtWJ.noalias() += Jt*Jt.transpose();
                mJtWr.noalias() -= rx*Jt;
                err_curr += rx*rx;
            }

            // JtWJ, JtWr for y
            Jt(0,0) = 0.0f;
            Jt(1,0) = fy*iz;
            Jt(2,0) =-fyyiz*iz;
            Jt(3,0) =-fy*(1.0f+yiz*yiz);
            Jt(4,0) = fyyiz*xiz;
            Jt(5,0) = fy*xiz;

             if(flag_weight) {
                float w_ry = weight*ry;
                float err = w_ry*ry;
                JtWJ.noalias() += weight*(Jt*Jt.transpose());
                mJtWr.noalias() -= (w_ry)*Jt;
                err_curr += err;
            }
            else {
                float err = ry*ry;
                JtWJ.noalias() += Jt*Jt.transpose();
                mJtWr.noalias() -= ry*Jt;
                err_curr += err;
            }
        } // END FOR
        err_curr *= inv_npts*0.5f;
        float delta_err = abs(err_curr-err_prev);

        // Solve H^-1*Jtr;
        for(int i = 0; i < 6; ++i) JtWJ(i,i) += lambda*JtWJ(i,i); // lambda 
        PoseSE3Tangent delta_xi = JtWJ.ldlt().solve(mJtWr);
        delta_xi *= step_size; 
        xi10 += delta_xi;
        err_prev = err_curr;
        // std::cout << "reproj. err. (avg): " << err_curr*inv_npts*0.5f << ", step: " << delta_xi.transpose() << std::endl;
        if(delta_xi.norm() < THRES_DELTA_XI || delta_err < THRES_DELTA_ERROR){
            std::cout << "poseonly BA stops at: " << iter <<", err: " << err_curr <<", derr: " << delta_err << ", # invalid: " << cnt_invalid << "\n";
            break;
        }
        if(iter == MAX_ITER-1){
            std::cout << "poseonly BA stops at full iterations!!" <<", err: " << err_curr <<", derr: " << delta_err << ", # invalid: " << cnt_invalid << "\n";
        }
    }

    if(!std::isnan(xi10.norm())){
        PoseSE3 T01_update;
        geometry::se3Exp_f(-xi10, T01_update);
        R01_true = T01_update.block<3,3>(0,0);
        t01_true = T01_update.block<3,1>(0,3);
    }
    else is_success = false;  // if nan, do not update.

    return is_success;
};

bool MotionEstimator::calcPoseOnlyBundleAdjustment(const LandmarkPtrVec& lms, const PixelVec& pts1, const std::shared_ptr<Camera>& cam,
    Rot3& Rwc_true, Pos3& twc_true, MaskVec& mask_inlier)
{
    // X is represented in the world frame.
    if(lms.size() != pts1.size()) 
        throw std::runtime_error("In 'calcPoseOnlyBundleAdjustment()': lms.size() != pts1.size().");
    
    bool is_success = true;

    int n_pts = lms.size();
    mask_inlier.resize(n_pts);
    
    int MAX_ITER = 250;
    float THRES_HUBER = 1.0f; // pixels
    float THRES_DELTA_XI = 1e-7;

    float lambda = 0.01f;
    float step_size = 1.0f;
    float THRES_REPROJ_ERROR = 3.0; // pixels
    
    float fx = cam->fx();
    float fy = cam->fy();
    float cx = cam->cx();
    float cy = cam->cy();
    float fxinv = cam->fxinv();
    float fyinv = cam->fyinv();

    PoseSE3 Tcw_init;
    PoseSE3 Twc_init;
    Twc_init << Rwc_true, twc_true, 0,0,0,1;
    Tcw_init = Tcw_init.inverse();

    PoseSE3Tangent xi10; // se3
    geometry::SE3Log_f(Tcw_init, xi10);
    
    for(uint32_t iter = 0; iter < MAX_ITER; ++iter){
        PoseSE3 Tcw;
        geometry::se3Exp_f(xi10,Tcw);

        Rot3 Rcw = Tcw.block<3,3>(0,0);
        Pos3 tcw = Tcw.block<3,1>(0,3);

        Eigen::Matrix<float,6,6> JtWJ;
        Eigen::Matrix<float,6,1> mJtWr;
        JtWJ.setZero();
        mJtWr.setZero();

        float err_prev = 1e15f;
        float err_curr = 0.0f;
        float inv_npts = 1.0f/(float)n_pts;
        // Warp and project point & calculate error...
        for(int i = 0; i < n_pts; ++i) {
            const Pixel& pt = pts1[i];
            Point Xprev = Rcw*lms[i]->get3DPoint() + tcw;

            float iz    = 1.0f/Xprev(2);
            float xiz   = Xprev(0)*iz;
            float yiz   = Xprev(1)*iz;
            float fxxiz = fx*xiz;
            float fyyiz = fy*yiz;

            Pixel pt_warp;
            pt_warp.x = fxxiz + cx;
            pt_warp.y = fyyiz + cy;

            float rx = pt_warp.x - pt.x;
            float ry = pt_warp.y - pt.y;
            
            // Huber weight calculation by the Manhattan distance
            float weight     = 1.0f;
            bool flag_weight = false;

            float absrxry = abs(rx)+abs(ry);
            if(absrxry >= THRES_HUBER){
                weight = THRES_HUBER/absrxry; 
                flag_weight = true;
            } 

            if(absrxry >= THRES_REPROJ_ERROR)
                mask_inlier[i] = false;
            else 
                mask_inlier[i] = true;

            // JtWJ, JtWr for x
            Eigen::Matrix<float,6,1> Jt;
            Jt(0,0) = fx*iz;
            Jt(1,0) = 0.0f;
            Jt(2,0) = -fxxiz*iz;
            Jt(3,0) = -fxxiz*yiz;
            Jt(4,0) = fx*(1.0f+xiz*xiz);
            Jt(5,0) = -fx*yiz;

            if(flag_weight) {
                float w_rx = weight*rx;
                float err = w_rx*rx;
                if(err <= THRES_REPROJ_ERROR){
                    mJtWr.noalias() -= (w_rx)*Jt;
                    JtWJ.noalias() += weight*(Jt*Jt.transpose());
                    err_curr += err;
                }
            }
            else {
                float err = rx*rx;
                if(err <= THRES_REPROJ_ERROR){
                    JtWJ.noalias() += Jt*Jt.transpose();
                    mJtWr.noalias() -= rx*Jt;
                    err_curr += rx*rx;
                }
            }

            // JtWJ, JtWr for y
            Jt(0,0) = 0.0f;
            Jt(1,0) = fy*iz;
            Jt(2,0) =-fyyiz*iz;
            Jt(3,0) =-fy*(1.0f+yiz*yiz);
            Jt(4,0) = fyyiz*xiz;
            Jt(5,0) = fy*xiz;

             if(flag_weight) {
                float w_ry = weight*ry;
                float err = w_ry*ry;
                if(err <= THRES_REPROJ_ERROR){
                    JtWJ.noalias() += weight*(Jt*Jt.transpose());
                    mJtWr.noalias() -= (w_ry)*Jt;
                    err_curr += err;
                }
            }
            else {
                float err = ry*ry;
                if(err <= THRES_REPROJ_ERROR){
                    JtWJ.noalias() += Jt*Jt.transpose();
                    mJtWr.noalias() -= ry*Jt;
                    err_curr += err;
                }
            }
        } // END FOR

        // Solve H^-1*Jtr;
        for(int i = 0; i < 6; ++i) JtWJ(i,i) += lambda*JtWJ(i,i); // lambda 
        PoseSE3Tangent delta_xi = JtWJ.ldlt().solve(mJtWr);
        delta_xi *= step_size; 
        xi10 += delta_xi;
        std::cout << "reproj. err. (avg): " << err_curr*inv_npts*0.5f << ", step: " << delta_xi.transpose() << std::endl;
        if(delta_xi.norm() < THRES_DELTA_XI){
            std::cout << "pose estimation stops at : " << iter <<"\n";
            break;
        }
    }

    if(!std::isnan(xi10.norm())){
        PoseSE3 T01_update;
        geometry::se3Exp_f(-xi10, T01_update);
        Rwc_true = T01_update.block<3,3>(0,0);
        twc_true = T01_update.block<3,1>(0,3);

        std::cout <<"BA result:\n";
        std::cout << "R01_ba:\n" << Rwc_true <<"\n";
        std::cout << "t01_ba:\n" << twc_true <<"\n";
    }
    else is_success = false;

    return is_success;
};

float MotionEstimator::calcSteeringAngleFromRotationMat(const Rot3& R){
    float psi = 0;
    Mat33 S = R-R.transpose();
    Vec3 v;
    v << -S(1,2),S(0,2),-S(0,1);

    Vec3 j_vec; j_vec << 0,1,0;
    float vjdot = v.dot(j_vec);

    psi = acos(0.5*(R.trace()-1.0f));
    if(vjdot < 0) psi = -psi;
    return psi;
};

bool MotionEstimator::localBundleAdjustment(const std::shared_ptr<Keyframes>& kfs, const std::shared_ptr<Camera>& cam)
{
    std::cout << "======= Local Bundle adjustment =======\n";

    int THRES_AGE          = 3;
    int THRES_MINIMUM_SEEN = 2;
    float THRES_PARALLAX   = 0.2*D2R;

    // Optimization paarameters
    int   MAX_ITER         = 250;

    float lam              = 1e-3;  // for Levenberg-Marquardt algorithm
    float MAX_LAM          = 1.0f;  // for Levenberg-Marquardt algorithm
    float MIN_LAM          = 1e-4f; // for Levenberg-Marquardt algorithm

    float THRES_HUBER      = 3.0f;
    float THRES_HUBER_MIN  = 0.5f;
    float THRES_HUBER_MAX  = 20.0f;

    bool DO_RECALC          = true;
    bool IS_DECREASE        = false;
    bool IS_STRICT_DECREASE = false;
    bool IS_BAD_DIVERGE     = false;


    if(kfs->getCurrentNumOfKeyframes() < 4){
        std::cout << "  -- Not enough keyframes... at least four keyframes are needed. local BA is skipped.\n";
        return false;
    }
    
    bool flag_success = true;

    // Make keyframe vector
    std::vector<FramePtr> kfs_all; // all keyframes
    for(auto kf : kfs->getList()) kfs_all.push_back(kf);
    std::vector<FramePtr> kfs_fixed; // fixed two frames
    kfs_fixed.push_back(kfs_all[0]);
    kfs_fixed.push_back(kfs_all[1]); // first two frames are fixed.
    std::cout << "# of all keyframes: " << kfs_all.size() << std::endl;

    // Landmark sets to generate observation graph
    std::vector<std::set<LandmarkPtr>> lmset_per_frame;
    std::set<LandmarkPtr> lmset_all;
    
    // 각 keyframe에서 보였던 lm을 저장. 단, age 가 3 이상인 경우만 포함.
    for(auto kf : kfs_all){
        std::set<LandmarkPtr> lmset_cur;
        for(auto lm : kf->getRelatedLandmarkPtr()){ 
            if(lm->getAge() >= THRES_AGE){ // age thresholding
                lmset_cur.insert(lm);
                lmset_all.insert(lm);
            }
        }
        lmset_per_frame.push_back(lmset_cur);
    }
    std::cout << " landmark set size: ";
    for(auto lmset : lmset_per_frame) std::cout << lmset.size() << " ";
    std::cout << "\n Unique landmarks: " << lmset_all.size() << std::endl;

    // 각 lm이 보였던 keyframe의 FramePtr을 저장.
    // 단, 최소 2개 이상의 keyframe 에서 관측되어야 local BA에서 사용함.    
    std::vector<LandmarkBA> lms_ba;
    for(auto lm : lmset_all){ // 모든 landmark를 순회.
        // 현재 landmark가 보였던 keyframes 중, 현재 active한 keyframe들만 추려냄. 
        LandmarkBA lm_ba;
        lm_ba.lm = lm;
        const FramePtrVec& kfs_related = lm->getRelatedKeyframePtr();
        const PixelVec& pts_on_kfs = lm->getObservationsOnKeyframes();
        
        for(int j = 0; j < kfs_related.size(); ++j){
            if(kfs_related[j]->isKeyframeInWindow()){
                lm_ba.kfs_seen.push_back(kfs_related[j]);
                lm_ba.pts_on_kfs.push_back(pts_on_kfs[j]);
            }
        }

        if(lm_ba.kfs_seen.size() >= THRES_MINIMUM_SEEN) lms_ba.push_back(lm_ba);
    }

    std::cout << "Landmark to be optimized: " << lms_ba.size() << std::endl;
    int n_obs = 0; // the number of total observations (2*n_obs == len_residual)
    for(auto lm_ba : lms_ba){
        // std::cout << lm_ba.lm->getID() <<" lm is recon?: " << lm_ba.lm->isTriangulated() << ", related to : ";
        for(auto kf : lm_ba.kfs_seen) std::cout << kf->getID() << " ";
        std::cout << std::endl;
        n_obs += lm_ba.kfs_seen.size();
    }

    // Intrinsic of lower camera
    Mat33 K = cam->K(); Mat33 Kinv = cam->Kinv();
    float fx = cam->fx(); float fy = cam->fy();
    float cx = cam->cx(); float cy = cam->cy();
    float invfx = cam->fyinv(); float invfy = cam->fxinv();

    // Parameter vector
    int N = kfs_all.size(); // the number of total frames
    int M = lms_ba.size(); // the number of total landmarks

    // initialize optimization parameter vector.
    int len_residual  = 2*n_obs;
    int len_parameter = 6*(N - 2) + 3*M;
    printf("======================================================\n");
    printf("Bundle Adjustment Statistics:\n");
    printf(" -        # of total images: %d images \n",    N);
    printf(" -        # of  opt. images: %d images \n",  N-2);
    printf(" -        # of total points: %d landmarks \n", M);
    printf(" -  # of total observations: %d \n", n_obs);
    printf(" -     Jacobian matrix size: %d rows x %d cols\n", len_residual, len_parameter);
    printf(" -     Residual vector size: %d rows\n", len_residual);


    // SE3 vector consisting of T_wj (world to a camera at each epoch)
    std::vector<PoseSE3> T_wj;         T_wj.reserve(N); // for optimization
    std::vector<PoseSE3> T_jw;         T_jw.reserve(N); // for optimization
    std::vector<PoseSE3Tangent> xi_jw; xi_jw.reserve(N); // for optimization
    for(int i = 0; i < N; ++i) {
        T_wj.push_back(kfs_all[i]->getPose());
        T_jw.push_back(T_wj[i].inverse());
        
        PoseSE3Tangent xi_jw_tmp;
        geometry::SE3Log_f(T_jw[i], xi_jw_tmp);
        xi_jw.push_back(xi_jw_tmp);
    }

    // initialize optimization parameter vector.
    SpVec parameter(len_parameter, 1);
    for(int i = 0; i < N-2; ++i){
        int idx0 = 6*i;
        PoseSE3Tangent xi_temp;
        geometry::SE3Log_f(T_jw[i+2], xi_temp);
        for(int j = 0; j < 6; ++j) {
            parameter.coeffRef(idx0+j,0) = xi_temp(j,0);
            // std::cout << "idx : " << idx0+j << std::endl;
        }
    }
    // X part (6*N + 3*m)~(6*N + 3*m + 2), ... ,(6*N + 3*(M-1))~(6*N + 3*(M-1) + 2)
    for(int i = 0; i < M; ++i){
        int idx0 = 6*(N-2) + 3*i;
        for(int j = 0; j < 3; ++j) {
            parameter.coeffRef(idx0+j,0) = lms_ba[i].lm->get3DPoint()(j);
            // std::cout << "idx : " << idx0+j << std::endl;
        }
    }

    // misc.
    std::vector<float> r_prev(n_obs, 0.0f);
    // for(int iter = 0; iter < MAX_ITER; ++iter) {       
    //     printf("ITERATION : %d\n", iter);
    //     // Generate parameters
    //     // xi part (0~5, 6~11, ... , 6*(N-1)~6*(N-1)+5)
    //     for(int i = 0; i < N; ++i){
    //         int idx0 = 6*i;
    //         se3 xi_temp;
    //         for(int j = 0; j < 6; ++j){
    //             xi_temp(j,0) = parameter.coeffRef(idx0+j,0);
    //             // std::cout << " j;" << j <<" : " << xi_temp(j,0) << std::endl;
    //         }
    //         SE3 T_tmp;
    //         sophuslie::se3Exp(  xi_temp,  T_tmp);
    //         T_w2l_inv[i] << T_tmp;
    //     }
    //     // X part (6*N + 3*m)~(6*N + 3*m + 2), ... ,(6*N + 3*(M-1))~(6*N + 3*(M-1) + 2)
    //     for(int i = 0; i < M; ++i){
    //         int idx0 = 6*N + 3*i;
    //         for(int j = 0; j < 3; ++j) db_valid[i]->X(j) = parameter.coeffRef(idx0+j,0);
    //     }

    //     // Generate Jacobian, Hessian, and residual.
    //     SpMat J(len_residual, len_parameter);
    //     SpVec r(len_residual, 1);
    //     SpTripletList Tplist;
    //     J.reserve(Eigen::VectorXi::Constant(len_parameter, 2000));

    //     // Set THRES_HUBER. 
    //     if(iter > 1){
    //         // printf("Recalculates THRES_HUBER.\n");
    //         std::sort(r_prev.begin(), r_prev.end(), [](int a, int b){return a < b;});
            
    //         THRES_HUBER = r_prev[(int)(0.6f*(float)r_prev.size())];
    //         if(THRES_HUBER > MAX_THRES_HUBER) THRES_HUBER = MAX_THRES_HUBER;
    //         if(THRES_HUBER < MIN_THRES_HUBER) THRES_HUBER = MIN_THRES_HUBER;
    //         printf("  -- THRES_HUBER is reset.: %0.3f [px]\n", THRES_HUBER);
    //         // sort(v.begin(), v.end(), [](int a, int b){ // 익명 함수를 이용한 compare
    //         //     if (a/10 > b/10){ // 왼쪽항의 십의 자리 수가 더 높다면
    //         //         return true; // 먼저 정렬한다.
    //         //     }
    //         //     else return a < b; // 그 외는 오름차순으로 정렬
    //         // });
    //     }

    //     // Calculate Jacobian and residual.
    //     int cnt = 0;
    //     for(int ii = 0; ii < M; ++ii){
    //         // image index where the i-th 3-D point is observed.
    //         const Point& Xi                = db_valid[ii]->X; // represented in the global frame.
    //         const Pixels& pts_l            = db_valid[ii]->pts_l;
    //         const Pixels& pts_u            = db_valid[ii]->pts_u;
    //         const std::vector<int>& id_kfs = db_valid[ii]->id_kfs;

    //         // printf("%d-th point = age: %ld\n",ii, db_valid[ii]->id_kfs.size());
    //         for(int jj = 0; jj < id_kfs.size(); ++jj){ 
    //             // One observation consists of onne point and one frame 
    //             // --> it generates four residual values.
    //             const Pixel& pt_l = pts_l[jj];
    //             const Pixel& pt_u = pts_u[jj];

    //             // Current image number
    //             const int& j = id_kfs[jj];

    //             const SE3& T_j2w = T_w2l_inv[j];
    //             SE3 T_uj2w  = T_ul*T_j2w;
    //             SO3 R_j2w   = T_j2w.block<3,3>(0,0);
    //             SO3 R_uj2w  = T_uj2w.block<3,3>(0,0);
    //             Vec3 t_j2w  = T_j2w.block<3,1>(0,3);
    //             Vec3 t_uj2w = T_uj2w.block<3,1>(0,3);

    //             Point Xi_lj = R_j2w *Xi    + t_j2w;
    //             Point Xi_uj =  R_ul *Xi_lj + t_ul;

    //             Eigen::Matrix3f hat_Xi_lj;
    //             hat_Xi_lj <<        0,-Xi_lj(2), Xi_lj(1), 
    //                          Xi_lj(2),        0,-Xi_lj(0),
    //                         -Xi_lj(1), Xi_lj(0),        0;

    //             const float& xlj = Xi_lj(0), ylj = Xi_lj(1), zlj = Xi_lj(2);
    //             const float& xuj = Xi_uj(0), yuj = Xi_uj(1), zuj = Xi_uj(2);
    //             // if(zlj < 1.0 || zuj < 1.0){
    //             //     std::cout << "zlj: " << zlj << ", zuj: " << zuj << std::endl;
    //             // }

    //             float invzlj = 1.0f/zlj; float invzlj2 = invzlj*invzlj;
    //             float invzuj = 1.0f/zuj; float invzuj2 = invzuj*invzuj;
                
    //             //     drudw = [fx_u*invzuj, 0,  -fx_u*xuj*invzuj2;...
    //             //         0, fy_u*invzuj,  -fy_u*yuj*invzuj2];
    //             Eigen::Matrix<float,2,3> drldw;
    //             drldw     << fx_l*invzlj, 0, -fx_l*xlj*invzlj2,
    //                          0, fy_l*invzlj, -fy_l*ylj*invzlj2;
                
    //             //     drudxi_jw = [drudw,-drudw*hat_Xi_uj];
    //             Eigen::Matrix<float,2,6> drldxi_jw;
    //             drldxi_jw << drldw, -drldw*hat_Xi_lj;
                
    //             //     drudX_i   = drudw*R_j2w;
    //             Eigen::Matrix<float,2,3> drldX_i;
    //             drldX_i   << drldw*R_j2w;

    //             //     drdw_l = [fx_l*invzlj, 0,  -fx_l*xlj*invzlj2;...
    //             //         0, fy_l*invzlj,  -fy_l*ylj*invzlj2];
    //             Eigen::Matrix<float,2,3> drudw;
    //             drudw     << fx_u*invzuj, 0, -fx_u*xuj*invzuj2,
    //                          0, fy_u*invzuj, -fy_u*yuj*invzuj2;
    //             //     drldxi_jw = drdw_l*[R_lu,-R_lu*hat_Xi_uj];
    //             Eigen::Matrix<float,2,6> drudxi_jw;
    //             Eigen::Matrix<float,2,3> drudwR_ul;
    //             drudwR_ul << drudw*R_ul;
    //             drudxi_jw << drudwR_ul, -drudwR_ul*hat_Xi_lj;
                
    //             //     drldX_i   = drdw_l*R_lj2w; // drldX_i   = drdw_l*R_lu*R_j2w;
    //             Eigen::Matrix<float,2,3> drudX_i;
    //             drudX_i << drudw*R_uj2w;

    //             // Fill Jacobian matrix!
    //             int j6 = j*6; int cnt4 = cnt*4;
    //             int idx_hori0, idx_hori1; 
    //             int idx_vert0_l, idx_vert1_l, idx_vert0_u, idx_vert1_u;
    //             idx_hori0 = j6; idx_hori1 = j6+5;
    //             idx_vert0_l = cnt4;   idx_vert1_l = cnt4+1;
    //             idx_vert0_u = cnt4+2; idx_vert1_u = cnt4+3;
    //             // 1) dru / dxi_jw
    //             // idx_hori = 6*(j-1)+(1:6); --> 6*(jj-1)+(0:5)
    //             // idx_vert = 4*(cnt-1)+(1:2); --> 4*(cnt-1)+0:1
    //             // % J(4*(cnt-1)+(1:2), 6*(j-1)+(1:6)) = drudxi_jw;
    //             // J(idx_vert, idx_hori) = drudxi_jw;
    //             this->fillTriplet(Tplist, idx_hori0, idx_hori1, idx_vert0_l, idx_vert1_l, drldxi_jw);

    //             // 2) drl / dxi_jw
    //             // % J(4*(cnt-1)+(3:4), 6*(j-1)+(1:6)) = drldxi_jw;
    //             // J(idx_vert+2, idx_hori) = drldxi_jw;
    //             this->fillTriplet(Tplist, idx_hori0, idx_hori1, idx_vert0_u, idx_vert1_u, drudxi_jw);

    //             // 3) dru / dX_i
    //             // % J(4*(cnt-1)+(1:2), 6*M+3*(ii-1)+(1:3)) = drudX_i;
    //             // idx_hori = 6*M+3*(ii-1)+(1:3);
    //             // J(idx_vert, idx_hori) = drudX_i;
    //             idx_hori0 = 6*N + 3*ii; 
    //             idx_hori1 = idx_hori0 + 2;
    //             this->fillTriplet(Tplist, idx_hori0, idx_hori1, idx_vert0_l, idx_vert1_l, drldX_i);
                
    //             // 4) drl / dX_i
    //             // % J(4*(cnt-1)+(3:4), 6*M+3*(ii-1)+(1:3)) = drldX_i;
    //             // J(idx_vert+2, idx_hori) = drldX_i;
    //             this->fillTriplet(Tplist, idx_hori0, idx_hori1, idx_vert0_u, idx_vert1_u, drudX_i);
                
    //             // 5) residual
    //             // ptsw_u = [fx_u*xuj*invzuj+cx_u; fy_u*yuj*invzuj+cy_u];
    //             // ptsw_l = [fx_l*xlj*invzlj+cx_l; fy_l*ylj*invzlj+cy_l];
    //             Pixel ptw_l, ptw_u;
    //             ptw_l.x = fx_l*xlj*invzlj + cx_l;
    //             ptw_l.y = fy_l*ylj*invzlj + cy_l;
    //             ptw_u.x = fx_u*xuj*invzuj + cx_u;
    //             ptw_u.y = fy_u*yuj*invzuj + cy_u;

    //             // r(4*(cnt-1)+(1:4),1) =...
    //             //     [ptsw_u-pts_u(:,jj);...
    //             //     ptsw_l-pts_l(:,jj)];
    //             r.coeffRef(cnt4  ,0) = ptw_l.x - pt_l.x;
    //             r.coeffRef(cnt4+1,0) = ptw_l.y - pt_l.y;
    //             r.coeffRef(cnt4+2,0) = ptw_u.x - pt_u.x;
    //             r.coeffRef(cnt4+3,0) = ptw_u.y - pt_u.y;

    //             ++cnt;
    //         } // END jj
    //     } // END ii 

    //     // Fill Jacobian
    //     // 'cnt' should be same with 'n_obs'
    //     // printf("cnt : %d, n_obs : %d\n", cnt, n_obs);
    //     size_t residual_size = 4*cnt;
    //     // std::cout << "iter : " << iter << ", # of Tplist: " << Tplist.size() <<"/" << 4*n_obs*(6*N+3*M) << ", percent: " << (float)Tplist.size() / (float)(len_parameter*len_residual)*100.0f << "%" << std::endl;
    //     J.setFromTriplets(Tplist.begin(), Tplist.end());
    //     J.makeCompressed();
    //     // std::cout << "residual nnz : " << r.nonZeros() <<" / " << residual_size << ", residual size : " << r.size() <<std::endl;

    //     // Huber loss calculation
    //     std::vector<float> err_per_db;
    //     std::vector<float> weight;
    //     err_per_db.resize(cnt);
    //     weight.resize(cnt);
    //     double err_sum = 0.0f;
    //     double* r_ptr = r.valuePtr();
    //     for(int ii = 0; ii < cnt; ++ii){
    //         float err_tmp = 0;
    //         int idx = 4*ii;
    //         const float u_l = r.coeff(idx  ,0); const float v_l = r.coeff(++idx,0);
    //         const float u_u = r.coeff(++idx,0); const float v_u = r.coeff(++idx,0);
    //         err_tmp += std::sqrt(u_l*u_l + v_l*v_l);
    //         err_tmp += std::sqrt(u_u*u_u + v_u*v_u);
    //         err_tmp *= 0.5f;

    //         // Huber calc.
    //         if(err_tmp > THRES_HUBER) weight[ii] = THRES_HUBER/err_tmp;
    //         else weight[ii] = 1.0f;

    //         r_prev[ii]     = err_tmp*weight[ii];
    //         err_per_db[ii] = err_tmp;
    //         err_sum += (double)(err_tmp*weight[ii]);
    //     }

    //     SpMat W(len_residual,len_residual);
    //     W.reserve(Eigen::VectorXi::Constant(len_residual,1));
    //     for(int ii = 0; ii < cnt; ++ii){
    //         int idx0 = 4*ii;
    //         float w_tmp = weight[ii];
    //         for(int jj = 0; jj < 4; ++jj) W.coeffRef(idx0 + jj, idx0 + jj) = w_tmp;
    //     }
    //     std::cout << "  iter: " << iter << ", errsum: " << err_sum <<", meanerr: " << err_sum/(float)cnt << " [px]" << std::endl;

    //     // Calc JtJ and Jtr
    //     // timer::tic();
    //     SpMat JtW = J.transpose()*W;
    //     SpMat JtWJ = JtW*J;
    //     for(int i = 0; i < JtWJ.cols(); ++i) JtWJ.coeffRef(i,i) *= (1.0f+lam); 

    //     // std::cout << "JtJ time : " << timer::toc() <<" [ms], ";
    //     // timer::tic();
    //     SpMat JtWr = JtW*r;
    //     // std::cout << "Jtr time : " << timer::toc() << " [ms], ";

    //     SpMat AA(6*N, 6*N);
    //     AA.reserve(Eigen::VectorXi::Constant(6*N, 6*N));
    //     SpMat BB(6*N, 3*M);
    //     BB.reserve(Eigen::VectorXi::Constant(6*N, 3*M));
    //     SpMat CC(3*M, 3*M);
    //     CC.reserve(Eigen::VectorXi::Constant(3*M,3));
        
    //     // SpMat CC(3*M, 3*M);
    //     // CC.reserve(Eigen::VectorXi::Constant(3*M,3));
        

    //     // Solve! (Cholesky decomposition based solver. JtJ is sym. positive definite.)
    //     // timer::tic();
    //     Eigen::SimplicialCholesky<SpMat> chol(JtWJ);
    //     Eigen::VectorXd delta_theta = chol.solve(JtWr);
    //     // std::cout << "chol time : " << timer::toc() << " [ms]" << std::endl;
    //     // std::cout << "dimension : " << delta_theta.rows() << ", 6*N+3*M: " << 6*N+3*M << std::endl;
        
    //     // Update parameters (T_w2l, T_w2l_inv, xi_w2l, database->X)
    //     // Should omit the first image pose update. (index = 0)
    //     double step_size = 0.2;
    //     SpVec delta_parameter(len_parameter, 1);
    //     for(int i = 0; i < 6; ++i) delta_parameter.coeffRef(i,0) = 0.0f;
    //     for(int i = 6; i < delta_theta.rows(); ++i) delta_parameter.coeffRef(i,0) = step_size*delta_theta(i);
        
    //     parameter -= delta_parameter; 

    // } // END iter


    // Finish
    std::cout << "======= Local Bundle adjustment - sucess:" << (flag_success ? "SUCCESS" : "FAILED") << "=======\n";
    return flag_success;
};