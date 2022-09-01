#include "core/motion_estimator.h"

MotionEstimator::MotionEstimator(){
    thres_1p_ = 10.0; // pixels
    thres_5p_ = 1.5; // pixels

    sparse_ba_solver_ = std::make_shared<SparseBundleAdjustmentSolver>();
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
    float THRES_HUBER        = 2.0f; // pixels
    float THRES_DELTA_XI     = 1e-7;
    float THRES_DELTA_ERROR  = 1e-5;
    float THRES_REPROJ_ERROR = 12.0f; // pixels

    float lambda = 0.001f;
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
    mask_inlier.resize(n_pts, true);
    
    int MAX_ITER = 250;
    float THRES_HUBER = 1.0f; // pixels
    float THRES_DELTA_XI = 1e-7;

    float lambda = 0.01f;
    float step_size = 1.0f;
    float THRES_REPROJ_ERROR = 4.0; // pixels
    float THRES_REPROJ_ERROR_INLIER = 3.0; // pixels
    
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

            if(iter > 2){
                if(absrxry >= THRES_REPROJ_ERROR_INLIER)
                    mask_inlier[i] = false;
                else 
                    mask_inlier[i] = true;
            }

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

    int THRES_AGE          = 3; // landmark의 최소 age
    int THRES_MINIMUM_SEEN = 2; // landmark의 최소 관측 keyframes
    float THRES_PARALLAX   = 0.3*D2R; // landmark의 최소 parallax

    // Optimization paarameters
    int   MAX_ITER         = 3;

    float lam              = 1e-3;  // for Levenberg-Marquardt algorithm
    float MAX_LAM          = 1.0f;  // for Levenberg-Marquardt algorithm
    float MIN_LAM          = 1e-4f; // for Levenberg-Marquardt algorithm

    float THRES_HUBER      = 1.5f;
    float THRES_HUBER_MIN  = 0.3f;
    float THRES_HUBER_MAX  = 20.0f;

    // optimization status flags
    bool DO_RECALC          = true;
    bool IS_DECREASE        = false;
    bool IS_STRICT_DECREASE = false;
    bool IS_BAD_DIVERGE     = false;

    float THRES_DELTA_THETA = 1e-7;
    float THRES_ERROR       = 1e-7;

    int NUM_MINIMUM_REQUIRED_KEYFRAMES = 5; // 최소 keyframe 갯수.
    int NUM_FIX_KEYFRAMES              = 3; // optimization에서 제외 할 keyframe 갯수. 과거 순.


    if(kfs->getCurrentNumOfKeyframes() < NUM_MINIMUM_REQUIRED_KEYFRAMES){
        std::cout << "  -- Not enough keyframes... at least four keyframes are needed. local BA is skipped.\n";
        return false;
    }
    
    bool flag_success = true;

    // Make keyframe vector
    std::vector<FramePtr> kfs_all; // all keyframes
    for(auto kf : kfs->getList()) kfs_all.push_back(kf);
    std::map<FramePtr,int> kfs_map;
    for(int i = 0; i < kfs_all.size(); ++i)
        kfs_map.insert(std::pair<FramePtr,int>(kfs_all[i],i));   
    std::cout << "# of all keyframes: " << kfs_all.size() << std::endl;
    for(auto kf : kfs_all) std::cout << kf->getID() << " ";
    std::cout << std::endl;

    // Landmark sets to generate observation graph
    std::vector<std::set<LandmarkPtr>> lmset_per_frame;
    std::set<LandmarkPtr> lmset_all;
    
    // 각 keyframe에서 보였던 lm을 저장. 단, age 가 3 이상인 경우만 포함.
    for(auto kf : kfs_all){
        std::set<LandmarkPtr> lmset_cur;
        for(auto lm : kf->getRelatedLandmarkPtr()){ 
            if(lm->getAge() >= THRES_AGE && lm->isTriangulated()){ // age thresholding
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
        lm_ba.X  = lm->get3DPoint();

        const FramePtrVec& kfvec_related = lm->getRelatedKeyframePtr();
        const PixelVec& pts_on_kfs     = lm->getObservationsOnKeyframes();
        for(int j = 0; j < kfvec_related.size(); ++j){
            if(kfvec_related[j]->isKeyframeInWindow()){ // optimization window내에 있는 keyframe인 경우.
                lm_ba.kfs_seen.push_back(kfvec_related[j]);
                lm_ba.kfs_index.push_back(kfs_map[kfvec_related[j]]);
                lm_ba.pts_on_kfs.push_back(pts_on_kfs[j]);
            }
        }

        if(lm_ba.kfs_seen.size() >= THRES_MINIMUM_SEEN){
            lms_ba.push_back(lm_ba); // minimum seen 을 넘긴 경우에만 optimization에 포함.
        }
    }

    std::cout << "Landmark to be optimized: " << lms_ba.size() << std::endl;
    int n_obs = 0; // the number of total observations (2*n_obs == len_residual)
    for(auto lm_ba : lms_ba){
        // for(auto kf : lm_ba.kfs_seen) std::cout << kf->getID() << " ";
        // std::cout << std::endl;
        n_obs += lm_ba.kfs_seen.size();
    }
    // 필요한 것. kfs_poses, lms_ba (Xw, pts_on_kfs, idx_kfs, ptr_kfs) 이렇게만 있으면 된다...

    // Intrinsic of lower camera
    Mat33 K = cam->K(); Mat33 Kinv = cam->Kinv();
    float fx = cam->fx(); float fy = cam->fy();
    float cx = cam->cx(); float cy = cam->cy();
    float invfx = cam->fxinv(); float invfy = cam->fyinv();

    // Parameter vector
    int N = kfs_all.size(); // the number of total frames
    int N_opt = N - NUM_FIX_KEYFRAMES; // the number of optimization frames
    int M = lms_ba.size(); // the number of total landmarks

    // initialize optimization parameter vector.
    int len_residual  = 2*n_obs;
    int len_parameter = 6*N_opt + 3*M;
    printf("======================================================\n");
    printf("Bundle Adjustment Statistics:\n");
    printf(" -        # of total images: %d images \n", N);
    printf(" -        # of  opt. images: %d images \n", N_opt);
    printf(" -        # of total points: %d landmarks \n", M);
    printf(" -  # of total observations: %d \n", n_obs);
    printf(" -     Jacobian matrix size: %d rows x %d cols\n", len_residual, len_parameter);
    printf(" -     Residual vector size: %d rows\n", len_residual);


    // SE3 vector consisting of T_wj (world to a camera at each epoch)
    std::vector<PoseSE3> T_wj_kfs(N); // for optimization
    std::vector<PoseSE3> T_jw_kfs(N); // for optimization
    std::vector<PoseSE3Tangent> xi_jw_kfs(N); // for optimization
    for(int j = 0; j < N; ++j) {
        T_wj_kfs[j] = kfs_all[j]->getPose();
        T_jw_kfs[j] = T_wj_kfs[j].inverse();
        
        PoseSE3Tangent xi_jw_tmp;
        geometry::SE3Log_f(T_jw_kfs[j], xi_jw_tmp);
        xi_jw_kfs[j] = xi_jw_tmp;
    }

    // initialize optimization parameter vector.
    SpVec parameter(len_parameter, 1);
    for(int j = 0; j < N_opt; ++j){
        int idx = 6*j;
        PoseSE3Tangent xi_jw;
        geometry::SE3Log_f(T_jw_kfs[j + NUM_FIX_KEYFRAMES], xi_jw);
        parameter.coeffRef(idx,0)   = xi_jw(0,0);
        parameter.coeffRef(++idx,0) = xi_jw(1,0);
        parameter.coeffRef(++idx,0) = xi_jw(2,0);
        parameter.coeffRef(++idx,0) = xi_jw(3,0);
        parameter.coeffRef(++idx,0) = xi_jw(4,0);
        parameter.coeffRef(++idx,0) = xi_jw(5,0);
    }

    // X part (6*N + 3*m)~(6*N + 3*m + 2), ... ,(6*N + 3*(M-1))~(6*N + 3*(M-1) + 2)
    for(int i = 0; i < M; ++i){
        int idx = 6*N_opt + 3*i;
        parameter.coeffRef(idx,0)   = lms_ba[i].X(0);
        parameter.coeffRef(++idx,0) = lms_ba[i].X(1);
        parameter.coeffRef(++idx,0) = lms_ba[i].X(2);
    }

    // misc.
    std::vector<float> r_prev(n_obs, 0.0f);
   
    // Preallocate Jacobian, Hessian, and residual
    // Vector generation & memory preallocation : 0.001 ms
    std::vector<Mat66>              Hpp_; // block diagonal. # of blocks: (N_opt)
    std::vector<std::vector<Mat63>> Hpl_; // large... # of blocks: N_opt X M
    std::vector<Mat33>              Hll_; // block diagonal, # of blocks: M
    std::vector<Mat33>              Hll_inv_; // block diagonal, # of blocks: M
    Hpp_.resize(N_opt);
    Hpl_.resize(N_opt); for(auto v : Hpl_) v.resize(M);
    Hll_.resize(M);
    Hll_inv_.resize(M);

    SpMat JtWJ(len_parameter,len_parameter);
    SpMat mJtWr(len_parameter,1);
    float err = 0.f;

    SpTripletList Tplist_JtWJ; Tplist_JtWJ.reserve(100000);
    SpTripletList Tplist_mJtWr; Tplist_mJtWr.reserve(50000); 
    JtWJ.reserve(Eigen::VectorXf::Constant(len_parameter,6+N*3));
    mJtWr.reserve(Eigen::VectorXf::Constant(len_parameter,1));
    for(int iter = 0; iter < MAX_ITER; ++iter){
        timer::tic();
        // Initialize JtWJ, mJtWr, err
        err = 0.0f;
        JtWJ.setZero();
        mJtWr.setZero();

        // Generate parameters
        // xi part 0~5, 6~11, ... , 6*(N-3)~6*(N-3)+5
        int idx = 0;
        for(int j = 0; j < N_opt; ++j){
            PoseSE3Tangent xi_jw;
            xi_jw(0,0) = parameter.coeff(idx++,0);
            xi_jw(1,0) = parameter.coeff(idx++,0);
            xi_jw(2,0) = parameter.coeff(idx++,0);
            xi_jw(3,0) = parameter.coeff(idx++,0);
            xi_jw(4,0) = parameter.coeff(idx++,0);
            xi_jw(5,0) = parameter.coeff(idx++,0);
            PoseSE3 T_jw;
            geometry::se3Exp_f(xi_jw, T_jw);
            T_jw_kfs[j+NUM_FIX_KEYFRAMES] = T_jw;
        }
        // point part 6*(N-2)~6*(N-2)+2, ... , 6*(N-2)+3*(M-1)~6*(N-2)+3*(M-1)+2
        idx = 6*N_opt;
        for(int i = 0; i < M; ++i){
            lms_ba[i].X(0) = parameter.coeff(idx++,0);
            lms_ba[i].X(1) = parameter.coeff(idx++,0);
            lms_ba[i].X(2) = parameter.coeff(idx++,0);
        }
        
        // Set THRES_HUBER
        if(iter > 1){
            // printf("Recalculates THRES_HUBER.\n");
            // float percentage_threshold = 0.6f;
            // std::sort(r_prev.begin(), r_prev.end(), [](int a, int b) {return a < b;});
            
            // THRES_HUBER = r_prev[(int)(percentage_threshold*(float)r_prev.size())];
            // if(THRES_HUBER > THRES_HUBER_MAX) THRES_HUBER = THRES_HUBER_MAX;
            // if(THRES_HUBER < THRES_HUBER_MIN) THRES_HUBER = THRES_HUBER_MIN;
            // printf("  -- THRES_HUBER is: %0.3f [px]\n", THRES_HUBER);
        }

        // Calculate Jacobian and residual.
        int cnt = 0;
        for(int i = 0; i < M; ++i){
            // image index where the i-th 3-D point is observed.
            const Point&            Xi      = lms_ba[i].X; // represented in the global frame.
            const PixelVec&         pts     = lms_ba[i].pts_on_kfs;
            const FramePtrVec&      kfs     = lms_ba[i].kfs_seen;
            const std::vector<int>& kfs_idx = lms_ba[i].kfs_index;
            
            for(int jj = 0; jj < kfs.size(); ++jj){ 
                const FramePtr& kf   = kfs[jj];
                const Pixel&    pij  = pts[jj];

                int j = kfs_idx[jj]; //  kf가 window 내에서 몇 번째 keyframe인지 나타냄.
                const PoseSE3&  T_jw = T_jw_kfs[j];

                // Current poses
                Rot3 R_jw = T_jw.block<3,3>(0,0);
                Pos3 t_jw = T_jw.block<3,1>(0,3);

                Point Xij = R_jw*Xi + t_jw; // transform a 3D point.

                // 1) Qij and Rij calculation.
                const float& xj = Xij(0), yj = Xij(1), zj = Xij(2);
                float invz = 1.0f/zj; float invz2 = invz*invz;
                
                float fxinvz      = fx*invz;
                float fyinvz      = fy*invz;
                float xinvz       = xj*invz;
                float yinvz       = yj*invz;
                float fx_xinvz2   = fxinvz*xinvz;
                float fy_yinvz2   = fyinvz*yinvz;
                float xinvz_yinvz = xinvz*yinvz;

                Mat23 Rij;
                const float& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2);
                const float& r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2);
                const float& r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
                Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                       fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33;


                Mat26 Qij;
                if(j >= NUM_FIX_KEYFRAMES){ // opt. frames.
                    Qij << fxinvz,0,-fx_xinvz2,-fx*xinvz_yinvz,fx*(1.f+xinvz*xinvz), -fx*yinvz,
                           0,fyinvz,-fy_yinvz2,-fy*(1.f+yinvz*yinvz),fy*xinvz_yinvz,  fy*xinvz;
                }

                // 2) residual calculation
                Vec2 rij;
                Pixel ptw;
                ptw.x = fx*xinvz + cx;
                ptw.y = fy*yinvz + cy;
                rij << ptw.x - pij.x, ptw.y - pij.y;

                // 3) HUBER weight calculation (Manhattan distance)
                float absrxry = abs(rij(0))+abs(rij(1));
                r_prev[cnt] = absrxry;

                float weight = 1.0f;
                bool flag_weight = false;
                if(absrxry > THRES_HUBER){
                    weight = (THRES_HUBER/absrxry);
                    flag_weight = true;
                }

                // 4) Add (or fill) data (JtWJ & mJtWr & err).      
                int idx_point0 = 6*N_opt + 3*i;
                int idx_point1 = idx_point0 + 2;

                if(j >= NUM_FIX_KEYFRAMES){ // opt. frames.
                    int idx_pose0 = 6*(j-NUM_FIX_KEYFRAMES);
                    int idx_pose1 = idx_pose0 + 5;
                    Mat66 Qij_t_Qij = Qij.transpose()*Qij; // fixed pose, opt. pose
                    Mat63 Qij_t_Rij = Qij.transpose()*Rij; // fixed pose, opt. pose
                    Mat36 Rij_t_Qij = Qij_t_Rij.transpose(); // fixed pose, opt. pose
                    Vec6  Qij_t_rij = Qij.transpose()*rij; // fixed pose, opt. pose
                    if(flag_weight){
                        Qij_t_Qij *= weight;
                        Qij_t_Rij *= weight;
                        Rij_t_Qij = Qij_t_Rij.transpose();
                        Qij_t_rij *= weight;
                    }
                    // JtWJ(idx_pose0:idx_pose1, idx_pose0:idx_pose1)  += weight*Qij.'*Qij;
                    // JtWJ(idx_pose0:idx_pose1, idx_point0:idx_point1) = weight*Qij.'*Rij;
                    // mJtWr(idx_pose0:idx_pose1,0)   -= weight*Qij.'*rij;
                    addData(JtWJ,    Qij_t_Qij, idx_pose0,  idx_pose0,  6,6);
                    insertData(JtWJ, Qij_t_Rij, idx_pose0,  idx_point0, 6,3);
                    insertData(JtWJ, Rij_t_Qij, idx_point0, idx_pose0,  3,6);          
                    addData(mJtWr,  -Qij_t_rij, idx_pose0,  0,          6,1);
                }

                Mat33 Rij_t_Rij = Rij.transpose()*Rij; // fixed pose
                Vec3 Rij_t_rij  = Rij.transpose()*rij; // fixed pose
                if(flag_weight){
                    Rij_t_Rij *= weight;
                    Rij_t_rij *= weight;
                }

                // JtWJ(idx_point0:idx_point1,idx_point0:idx_point1) += weight*Rij.'*Rij;
                // mJtWr(idx_point0:idx_point1,0) -= weight*Rij.'*rij;
                addData(JtWJ, Rij_t_Rij, idx_point0, idx_point0, 3,3);
                addData(mJtWr,-Rij_t_rij, idx_point0, 0, 3,1);

                float err_tmp = weight*rij.transpose()*rij;
                err += err_tmp;
               
                ++cnt;
            } // END j
        } // END i 
        timer::toc(1);

        // Fill Jacobian
        // 'cnt' should be same with 'n_obs'
        // printf("cnt : %d, n_obs : %d\n", cnt, n_obs);
        timer::tic();
        int residual_size = 2*cnt;
        // std::cout << "iter : " << iter << ", # of Tplist: " << Tplist.size() <<"/" << 4*n_obs*(6*N+3*M) << ", percent: " << (float)Tplist.size() / (float)(len_parameter*len_residual)*100.0f << "%" << std::endl;
        JtWJ.makeCompressed();
        mJtWr.makeCompressed();
        // std::cout << "residual nnz : " << r.nonZeros() <<" / " << residual_size << ", residual size : " << r.size() <<std::endl;
        timer::toc(1);

        // Damping (lambda)
        for(int i = 0; i < JtWJ.cols(); ++i) JtWJ.coeffRef(i,i) *= (1.0f+lam); 

        timer::tic();
        SpMat AA(6*N_opt, 6*N_opt);
        SpMat BB(6*N_opt, 3*M);
        SpMat CC(3*M,     3*M);
        AA.reserve(Eigen::VectorXf::Constant(6*N_opt, 6));
        BB.reserve(Eigen::VectorXf::Constant(6*N_opt, 3*N));
        CC.reserve(Eigen::VectorXf::Constant(3*M,3));
        AA = JtWJ.block(0,0, 6*N_opt, 6*N_opt);
        BB = JtWJ.block(0,6*N_opt, 6*N_opt, 3*M);
        CC = JtWJ.block(6*N_opt,6*N_opt,3*M,3*M);
        for(int i = 0; i < N_opt; ++i){
            int idx = 3*i;
            Mat33 C_tmp = CC.block(idx,idx,3,3);
            C_tmp = C_tmp.inverse();
            CC.coeffRef(idx  ,idx  ) = C_tmp(0,0);
            CC.coeffRef(idx  ,idx+1) = C_tmp(0,1);
            CC.coeffRef(idx  ,idx+2) = C_tmp(0,2);
            CC.coeffRef(idx+1,idx  ) = C_tmp(1,0);
            CC.coeffRef(idx+1,idx+1) = C_tmp(1,1);
            CC.coeffRef(idx+1,idx+2) = C_tmp(1,2);
            CC.coeffRef(idx+2,idx  ) = C_tmp(2,0);
            CC.coeffRef(idx+2,idx+1) = C_tmp(2,1);
            CC.coeffRef(idx+2,idx+2) = C_tmp(2,2);
        }
        // SpMat BCinv = BB*CC;
        timer::toc(1);

        // SpMat CC(3*M, 3*M);
        // CC.reserve(Eigen::VectorXi::Constant(3*M,3));
        
        // Solve! (Cholesky decomposition based solver. JtJ is sym. positive definite.)
        // timer::tic();
        // SpMat a = mJtWr.block(0,0,6*N_opt,1);
        // SpMat b = mJtWr.block(6*N_opt,0,3*M,1);
        // Eigen::SimplicialCholesky<SpMat> chol11(AA-BCinv*BB.transpose());
        // Eigen::VectorXf x = chol11.solve(a-BCinv*b);
        // Eigen::VectorXf y = CC*b-BCinv.transpose()*x;
        // Eigen::VectorXf delta_theta;
        // delta_theta << x, y;
        // timer::toc(1);
        timer::tic();
        Eigen::SimplicialCholesky<SpMat> chol(JtWJ);
        Eigen::VectorXf  delta_theta = chol.solve(mJtWr);
        // std::cout << delta_theta.transpose() <<std::endl;
        std::cout << "chol time : " << timer::toc(0) << " [ms]" << std::endl; // 그냥 통째로 풀면 한 iteration 당 40 ms (desktop)

        // std::cout << "dimension : " << delta_theta.rows() << ", 6*N+3*M: " << 6*N+3*M << std::endl;
        
        // Update parameters (T_w2l, T_w2l_inv, xi_w2l, database->X)
        // Should omit the first image pose update. (index = 0)
        double step_size = 1.0;
        SpVec delta_parameter(len_parameter, 1);
        for(int i = 0; i < delta_theta.rows(); ++i) delta_parameter.coeffRef(i,0) = delta_theta(i);
        std::cout << "  iter: " << iter << ", meanerr: " << err/(float)cnt << " [px], delta xi: " << delta_theta.block<12,1>(0,0).norm() << std::endl;

        parameter += delta_parameter; 
    } // END iter

    // update parameters
    for(int j = NUM_FIX_KEYFRAMES; j < kfs_all.size(); ++j){
        kfs_all[j]->setPose(T_jw_kfs[j].inverse());
    }
    for(int i = 0; i < lms_ba.size(); ++i){
        lms_ba[i].lm->set3DPoint(lms_ba[i].X);
    }

    // Finish
    std::cout << "======= Local Bundle adjustment - sucess:" << (flag_success ? "SUCCESS" : "FAILED") << "=======\n";
    return flag_success;
};

void MotionEstimator::addData(SpMat& mat, const Eigen::MatrixXf& mat_part, int row_start, int col_start, int row_sz, int col_sz)
{
    // Sparse matrix default : column-major order. 
    for(int j = 0; j < col_sz; ++j){
        int col_mat = j+col_start;
        for(int i = 0; i < row_sz; ++i){
            mat.coeffRef(i+row_start,col_mat) += mat_part(i,j);
        }
    }
}
void MotionEstimator::insertData(SpMat& mat, const Eigen::MatrixXf& mat_part, int row_start, int col_start, int row_sz, int col_sz)
{
    // Sparse matrix default : column-major order. 
    for(int j = 0; j < col_sz; ++j){
        int col_mat = j+col_start;
        for(int i = 0; i < row_sz; ++i){
            mat.insert(i+row_start,col_mat) = mat_part(i,j);
        }
    }
}

inline void MotionEstimator::fillTriplet(SpTripletList& Tri, const int& idx_hori0, const int& idx_hori1, 
    const int& idx_vert0, const int& idx_vert1, const Eigen::MatrixXf& mat)
{
    int dim_hori = idx_hori1 - idx_hori0 + 1;
    int dim_vert = idx_vert1 - idx_vert0 + 1;

    if(mat.cols() != dim_hori) throw std::runtime_error("BundleAdjustmentSolver::fillJacobian(...), mat.cols() != dim_hori\n");
    if(mat.rows() != dim_vert) throw std::runtime_error("BundleAdjustmentSolver::fillJacobian(...), mat.rows() != dim_vert\n");

    for(int u = 0; u < dim_hori; ++u){
        for(int v = 0; v < dim_vert; ++v){
            Tri.push_back(SpTriplet(v + idx_vert0, u + idx_hori0, mat(v,u)));
        }
    }
};

bool MotionEstimator::localBundleAdjustmentSparseSolver(const std::shared_ptr<Keyframes>& kfs_window, const std::shared_ptr<Camera>& cam)
{
// Variables
// 
// LandmarkBAVec lms_ba; 
//  - Xi
//  - pts_on_kfs
//  - kfs_seen
// 
// std::map<FramePtr,int>     kfmap_optimizable
// std::map<FramePtr,PoseSE3> Tjw_map;

    std::cout << "===================== Local Bundle adjustment2 ============================\n";

    int THRES_AGE           = 2; // landmark의 최소 age
    int THRES_MINIMUM_SEEN  = 2; // landmark의 최소 관측 keyframes
    float THRES_PARALLAX    = 0.3*D2R; // landmark의 최소 parallax

    // Optimization paraameters
    int   MAX_ITER          = 10;

    float lam               = 1e-3;  // for Levenberg-Marquardt algorithm
    float MAX_LAM           = 1.0f;  // for Levenberg-Marquardt algorithm
    float MIN_LAM           = 1e-4f; // for Levenberg-Marquardt algorithm

    float THRES_HUBER       = 1.5f;
    float THRES_HUBER_MIN   = 0.3f;
    float THRES_HUBER_MAX   = 20.0f;

    int   THRES_NUM_MAXIMUM_PAST_KEYFRAME_ID = 15;

    // optimization status flags
    bool DO_RECALC          = true;
    bool IS_DECREASE        = false;
    bool IS_STRICT_DECREASE = false;
    bool IS_BAD_DIVERGE     = false;

    float THRES_DELTA_THETA = 1e-7;
    float THRES_ERROR       = 1e-7;

    int NUM_MINIMUM_REQUIRED_KEYFRAMES = 4; // 최소 keyframe 갯수.
    int NUM_FIX_KEYFRAMES_IN_WINDOW    = 3; // optimization에서 제외 할 keyframe 갯수. 과거 순.

    // Check whether there are enough keyframes
    if(kfs_window->getCurrentNumOfKeyframes() < NUM_MINIMUM_REQUIRED_KEYFRAMES){
        std::cout << "  -- Not enough keyframes... at least four keyframes are needed. local BA is skipped.\n";
        return false;
    }

    // Do Local Bundle Adjustment.
    bool flag_success = true; // Local BA success flag.

    // make opt and non opt images
    FramePtrVec frames;
    std::vector<int> idx_fix;
    std::vector<int> idx_opt;
    for(auto kf : kfs_window->getList()) // 모든 keyframe in window 순회 
        frames.push_back(kf); // window keyframes 저장.
    
    for(int j = 0; j < NUM_FIX_KEYFRAMES_IN_WINDOW; ++j)
        idx_fix.push_back(j);
    for(int j = NUM_FIX_KEYFRAMES_IN_WINDOW; j < frames.size(); ++j)
        idx_opt.push_back(j);
        
    std::shared_ptr<SparseBAParameters> ba_params;
    ba_params = std::make_shared<SparseBAParameters>();
    ba_params->setPosesAndPoints(frames, idx_fix, idx_opt);

    // BA solver.
    timer::tic();
    sparse_ba_solver_->reset();
    sparse_ba_solver_->setCamera(cam);
    sparse_ba_solver_->setBAParameters(ba_params);
    sparse_ba_solver_->setHuberThreshold(THRES_HUBER);
    std::cout << "LBA time to prepare: " << timer::toc(0) << " [ms]\n";

    timer::tic();
    sparse_ba_solver_->solveForFiniteIterations(MAX_ITER);
    std::cout << "LBA time to solve: " << timer::toc(0) << " [ms]\n";

    timer::tic();
    sparse_ba_solver_->reset();
    std::cout << "LBA time to reset: "<< timer::toc(0) << " [ms]\n";

    return true;
};
