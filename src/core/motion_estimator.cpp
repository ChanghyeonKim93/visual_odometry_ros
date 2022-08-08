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