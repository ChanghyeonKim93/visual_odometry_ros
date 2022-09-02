#include "core/ba_solver/sparse_bundle_adjustment.h"
/*
    <Problem we want to solve>
        H*delta_theta = J.transpose()*r;

    where
    - Hessian  
            H = [A_,  B_;
                 Bt_, C_];

    - Jacobian multiplied by residual vector 
            J.transpose()*r = [a;b];

    - Update parameters
            delta_theta = [x;y];
*/
// A sparse solver for a feature-based Bundle adjustment problem.
SparseBundleAdjustmentSolver::SparseBundleAdjustmentSolver() 
: N_(0), N_opt_(0), M_(0), n_obs_(0), THRES_EPS_(0), THRES_HUBER_(0)
{
    cam_       = nullptr;
    ba_params_ = nullptr;

    A_.reserve(500); // reserve expected # of optimizable poses (N_opt)
    B_.reserve(500); // 
    Bt_.reserve(5000);
    C_.reserve(5000); // reserve expected # of optimizable landmarks (M)
};

// Set connectivities, variables...
void SparseBundleAdjustmentSolver::setBAParameters(const std::shared_ptr<SparseBAParameters>& ba_params)
{
    ba_params_ = ba_params;
    
    N_     = ba_params_->getNumOfAllFrames();
    N_opt_ = ba_params_->getNumOfOptimizeFrames();
    M_     = ba_params_->getNumOfOptimizeLandmarks();
    n_obs_ = ba_params_->getNumOfObservations();

    this->setProblemSize(N_, N_opt_, M_, n_obs_);
};

// Set Huber threshold
void SparseBundleAdjustmentSolver::setHuberThreshold(float thres_huber){
    THRES_HUBER_ = thres_huber;
};

// Set camera.
void SparseBundleAdjustmentSolver::setCamera(const std::shared_ptr<Camera>& cam){
    cam_ = cam;
};

// Set problem sizes and resize the storages.
void SparseBundleAdjustmentSolver::setProblemSize(int N, int N_opt, int M, int n_obs){ 
    // Set sizes
    N_     = N; // # of total keyframes (including fixed frames)
    N_opt_ = N_opt; // # of optimizable keyframes
    M_     = M; // # of landmarks to be optimized.
    n_obs_ = n_obs;

    // Resize storages.
    A_.resize(N_opt_); 
    
    B_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j) 
        B_[j].resize(M_, Mat63::Zero());   // 6x3, N_opt X M blocks
    
    Bt_.resize(M_);    
    for(int i = 0; i < M_; ++i) 
        Bt_[i].resize(N_opt_, Mat36::Zero());   // 3x6, N_opt X M blocks
    
    C_.resize(M_);

    a_.resize(N_opt_); // 6x1, N_opt blocks
    x_.resize(N_opt_); // 6x1, N_opt blocks
    params_poses_.resize(N_opt_); // 6x1, N_opt blocks

    b_.resize(M_);     // 3x1, M blocks
    y_.resize(M_);     // 3x1, M blocks
    params_points_.resize(M_);    // 3x1, M blocks

    Cinv_.resize(M_); // 3x3, M diagonal blocks 

    BCinv_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        BCinv_[j].resize(M_, Mat63::Zero());   // 6x3, N_opt X M blocks
    
    CinvBt_.resize(M_); 
    for(int i = 0 ; i < M_; ++i) 
        CinvBt_[i].resize(N_opt_,Mat36::Zero());
    
    BCinvBt_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        BCinvBt_[j].resize(N_opt_, Mat66::Zero());   // 6x6, N_opt X N_opt blocks
    
    BCinv_b_.resize(N_opt_); // 6x1, N_opt x 1 blocks
    am_BCinv_b_.resize(N_opt_); // 6x1, N_opt x 1 blocks

    Am_BCinvBt_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        Am_BCinvBt_[j].resize(N_opt_, Mat66::Zero());   // 6x6, N_opt X N_opt blocks
    
    Cinv_b_.resize(M_);
    Bt_x_.resize(M_);        // 3x1, M x 1 blocks
    CinvBt_x_.resize(M_);

};

// Solve the BA for fixed number of iterations
bool SparseBundleAdjustmentSolver::solveForFiniteIterations(int MAX_ITER){
    
    bool flag_nan_pass   = true;
    bool flag_error_pass = true;
    bool flag_success    = true;

    float THRES_SUCCESS_AVG_ERROR = 1.0f;

    float MAX_LAMBDA = 20.0f;
    float MIN_LAMBDA = 1e-6f;

    // Intrinsic of lower camera
    const Mat33& K = cam_->K(); const Mat33& Kinv = cam_->Kinv();
    const float& fx = cam_->fx(); const float& fy = cam_->fy();
    const float& cx = cam_->cx(); const float& cy = cam_->cy();
    const float& invfx = cam_->fxinv(); const float& invfy = cam_->fyinv();

    // Set the Parameter Vector.
    setParameterVectorFromPosesPoints();            

    // Initialize parameters
    std::vector<float> r_prev(n_obs_, 0.0f);
    float err_prev = 1e10f;
    float lambda = 0.00000000001;
    for(int iter = 0; iter < MAX_ITER; ++iter){
        // set Poses and Points.
        getPosesPointsFromParameterVector();

        // Reset A, B, Bt, C, Cinv, a, b, x, y...
        zeroizeStorageMatrices();

        // Iteratively solve. (Levenberg-Marquardt algorithm)
        int cnt = 0;
        float err = 0.0f;
        for(int i = 0; i < M_; ++i){
            // For i-th landmark
            const LandmarkBA&  lmba = ba_params_->getLandmarkBA(i);
            const Point&       Xi   = lmba.X; 
            const FramePtrVec& kfs  = lmba.kfs_seen;
            const PixelVec&    pts  = lmba.pts_on_kfs;

            for(int jj = 0; jj < kfs.size(); ++jj){
                // For j-th keyframe
                const Pixel&   pij = pts[jj];
                const FramePtr& kf = kfs[jj];

                // 0) check whether it is optimizable keyframe
                bool is_optimizable_keyframe = false;
                int j = -1;
                if(ba_params_->isOptFrame(kf)){
                    is_optimizable_keyframe = true;
                    j = ba_params_->getOptPoseIndex(kf);
                }

                // Current poses
                const PoseSE3& T_jw = ba_params_->getPose(kf);
                const Rot3&    R_jw = T_jw.block<3,3>(0,0);
                const Pos3&    t_jw = T_jw.block<3,1>(0,3);
                
                Point Xij = R_jw*Xi + t_jw; // transform a 3D point.
            
                // 1) Qij and Rij calculation.
                const float& xj = Xij(0), yj = Xij(1), zj = Xij(2);
                float invz = 1.0f/zj; float invz2 = invz*invz;
                
                float fxinvz      = fx*invz;      float fyinvz      = fy*invz;
                float xinvz       = xj*invz;      float yinvz       = yj*invz;
                float fx_xinvz2   = fxinvz*xinvz; float fy_yinvz2   = fyinvz*yinvz;
                float xinvz_yinvz = xinvz*yinvz;

                Mat23 Rij;
                const float& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2);
                const float& r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2);
                const float& r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
                Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                       fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33;

                // 2) residual calculation
                Vec2 rij;
                Pixel ptw;
                ptw.x = fx*xinvz + cx; ptw.y = fy*yinvz + cy;
                rij << ptw.x - pij.x, ptw.y - pij.y;

                // 3) HUBER weight calculation (Manhattan distance)
                float absrxry = abs(rij(0))+abs(rij(1));
                r_prev[cnt] = absrxry;
                // std::cout << cnt << "-th point: " << absrxry << " [px]\n";

                float weight = 1.0f;
                bool flag_weight = false;
                if(absrxry > THRES_HUBER_){
                    weight = (THRES_HUBER_/absrxry);
                    flag_weight = true;
                }

                // 4) Add (or fill) data (JtWJ & mJtWr & err).  
                Mat33 Rij_t_Rij = Rij.transpose()*Rij; // fixed pose
                Vec3  Rij_t_rij = Rij.transpose()*rij; // fixed pose
                if(flag_weight){
                    Rij_t_Rij *= weight;
                    Rij_t_rij *= weight;
                }

                C_[i]   += Rij_t_Rij; // FILL STORAGE (3)
                b_[i]   -= Rij_t_rij; // FILL STORAGE (5)

                if(is_optimizable_keyframe) { // Optimizable keyframe.
                    Mat26 Qij;
                    Qij << fxinvz,0,-fx_xinvz2,-fx*xinvz_yinvz,fx*(1.f+xinvz*xinvz), -fx*yinvz,
                           0,fyinvz,-fy_yinvz2,-fy*(1.f+yinvz*yinvz),fy*xinvz_yinvz,  fy*xinvz;
                           
                    Mat66 Qij_t_Qij = Qij.transpose()*Qij; // fixed pose, opt. pose
                    Mat63 Qij_t_Rij = Qij.transpose()*Rij; // fixed pose, opt. pose
                    Vec6 Qij_t_rij = Qij.transpose()*rij; // fixed pose, opt. pose
                    if(flag_weight){
                        Qij_t_Qij *= weight;
                        Qij_t_Rij *= weight;
                        Qij_t_rij *= weight;
                    }

                    A_[j]    += Qij_t_Qij; // FILL STORAGE (1)
                    B_[j][i]  = Qij_t_Rij; // FILL STORAGE (2)
                    Bt_[i][j] = Qij_t_Rij.transpose(); // FILL STORAGE (2-1)
                    a_[j]    -= Qij_t_rij; // FILL STORAGE (4)
                } 
                float err_tmp = weight*rij.transpose()*rij;
                err += err_tmp;

                ++cnt;
            } // END jj

            // Damping term
            for(int j = 0; j < N_opt_; ++j){
                A_[j] += lambda*Mat66::Identity();
                // A_[j](0,0) *= (1.0f+lambda);
                // A_[j](1,1) *= (1.0f+lambda);
                // A_[j](2,2) *= (1.0f+lambda);
                // A_[j](3,3) *= (1.0f+lambda);
                // A_[j](4,4) *= (1.0f+lambda);
                // A_[j](5,5) *= (1.0f+lambda);
            }
            for(int i = 0; i < M_; ++i){
                C_[i] += lambda*Mat33::Identity();
                // C_[i](0,0) *= (1.0f+lambda);
                // C_[i](1,1) *= (1.0f+lambda);
                // C_[i](2,2) *= (1.0f+lambda);
            }

            Cinv_[i]   = C_[i].inverse(); // FILL STORAGE (3-1)
            Cinv_b_[i] = Cinv_[i]*b_[i];  // FILL STORAGE (10)
            for(int jj = 0; jj < kfs.size(); ++jj){
                // For j-th keyframe
                const FramePtr& kf = kfs[jj];

                // 0) check whether it is optimizable keyframe
                bool is_optimizable_keyframe_j = false;
                int j = -1;
                if(ba_params_->isOptFrame(kf)){
                    is_optimizable_keyframe_j = true;
                    j = ba_params_->getOptPoseIndex(kf);

                    BCinv_[j][i]  = B_[j][i]*Cinv_[i];  // FILL STORAGE (6)
                    CinvBt_[i][j] = BCinv_[j][i].transpose(); // FILL STORAGE (11)
                    BCinv_b_[j]  += BCinv_[j][i]*b_[i];  // FILL STORAGE (9)
                }

                for(int kk = jj; kk < kfs.size(); ++kk){
                    // For k-th keyframe
                    const FramePtr& kf2 = kfs[kk];
                    bool is_optimizable_keyframe_k = false;
                    int k = -1;
                    if(ba_params_->isOptFrame(kf2)){
                        is_optimizable_keyframe_k = true;
                        k = ba_params_->getOptPoseIndex(kf2);
                    }

                    if(is_optimizable_keyframe_j && is_optimizable_keyframe_k){
                        BCinvBt_[j][k] += BCinv_[j][i]*Bt_[i][k];  // FILL STORAGE (7)
                    }
                }
            } // END jj
        } // END i

        for(int j = 0; j < N_opt_; ++j)
            for(int k = j; k < N_opt_; ++k)
                BCinvBt_[k][j] = BCinvBt_[j][k].transpose();
            
        for(int j = 0; j < N_opt_; ++j){
            for(int k = 0; k < N_opt_; ++k){
                if(j == k) Am_BCinvBt_[j][k] = A_[j] - BCinvBt_[j][k];
                else       Am_BCinvBt_[j][k] =       - BCinvBt_[j][k];
            }
        }
        for(int j = 0; j < N_opt_; ++j) 
            am_BCinv_b_[j] = a_[j] - BCinv_b_[j];
        
        // Solve problem.
        // 1) solve x
        Eigen::MatrixXf Am_BCinvBt_mat(6*N_opt_,6*N_opt_);
        Eigen::MatrixXf am_BCinv_b_mat(6*N_opt_,1);
        
        for(int j = 0; j < N_opt_; ++j){
            int idx0 = 6*j;
            for(int k = 0; k < N_opt_; ++k){
                int idx1 = 6*k;
                Am_BCinvBt_mat.block(idx0,idx1,6,6) = Am_BCinvBt_[j][k];
            }
            am_BCinv_b_mat.block(idx0,0,6,1) = am_BCinv_b_[j];
        }
        Eigen::MatrixXf x_mat = Am_BCinvBt_mat.ldlt().solve(am_BCinv_b_mat);
        
        for(int j = 0; j < N_opt_; ++j)
            x_[j] = x_mat.block<6,1>(6*j,0);

        for(int i = 0; i < M_; ++i){
            const LandmarkBA& lmba = ba_params_->getLandmarkBA(i);
            const FramePtrVec& kfs = lmba.kfs_seen;
            for(int jj = 0; jj < kfs.size(); ++jj){
                const FramePtr& kf = kfs[jj];

                if(ba_params_->isOptFrame(kf)){
                    int j = ba_params_->getOptPoseIndex(kf);
                    CinvBt_x_[i] += CinvBt_[i][j]*x_[j];
                }
            }
            y_[i] = Cinv_b_[i] - CinvBt_x_[i];
        }
                
        // Update step
        for(int j_opt = 0; j_opt < N_opt_; ++j_opt){
            // PoseSE3 Tjw_update, dT;
            // geometry::se3Exp_f(params_poses_[j_opt],Tjw_update);
            // geometry::se3Exp_f(x_[j_opt],dT);
            // Tjw_update = dT*Tjw_update;
            // geometry::SE3Log_f(Tjw_update,params_poses_[j_opt]);
            geometry::addFrontse3_f(params_poses_[j_opt], x_[j_opt]);
            // params_poses_[j_opt] += x_[j_opt];
        }

        for(int i = 0; i < M_; ++i)
            params_points_[i] += y_[i];

        float average_error = 0.5f*err/(float)n_obs_;
            
        std::cout << iter << "-th iter, error : " << average_error << "\n";

        // Check extraordinary cases.
        flag_nan_pass   = std::isnan(err) ? false : true;
        flag_error_pass = (average_error <= THRES_SUCCESS_AVG_ERROR) ? true : false;
        flag_success    = flag_nan_pass && flag_error_pass;
    } // END iter

    // Finally, update parameters
    if(flag_nan_pass && flag_error_pass){
        for(int j_opt = 0; j_opt < N_opt_; ++j_opt){
            const FramePtr& kf = ba_params_->getOptFramePtr(j_opt);
            const PoseSE3& Tjw_update = ba_params_->getPose(kf);
            kf->setPose(Tjw_update.inverse());
        }
        for(int i = 0; i < M_; ++i){
            const LandmarkBA& lmba = ba_params_->getLandmarkBA(i);
            const LandmarkPtr& lm = lmba.lm;

            lm->set3DPoint(lmba.X);
            lm->setBundled();
        }
    }
    else{
        std::cout << "************************* LOCAL BA FAILED!!!!! ****************************\n";
        std::cout << "************************* LOCAL BA FAILED!!!!! ****************************\n";
        std::cout << "************************* LOCAL BA FAILED!!!!! ****************************\n";
    }

    // Finish
    // std::cout << "======= Local Bundle adjustment - sucess:" << (flag_success ? "SUCCESS" : "FAILED") << "=======\n";

    return flag_success;
};

// Reset local BA solver.
void SparseBundleAdjustmentSolver::reset(){
    ba_params_ = nullptr;
    cam_       = nullptr;
    N_ = 0;
    N_opt_ = 0;
    M_ = 0;
    n_obs_ = 0;
    THRES_EPS_ = 0;
    THRES_HUBER_ = 0;

    A_.resize(0);
    B_.resize(0);
    Bt_.resize(0);
    C_.resize(0);

    a_.resize(0);
    b_.resize(0);

    Cinv_.resize(0); // M blocks diagonal part for landmarks' 3D points (inverse)
    BCinv_.resize(0); // N_opt X M blocks (6x3)
    CinvBt_.resize(0);
    BCinvBt_.resize(0); // N_opt x N_opt blocks (6x6)
    BCinv_b_.resize(0); // N_opt x 1 blocks (6x1)
    Bt_x_.resize(0); // M x 1 blocks (3x1)

    x_.resize(0); // N_opt blocks (6x1)
    y_.resize(0); // M blocks (3x1)

    params_poses_.resize(0);
    params_points_.resize(0);

    Am_BCinvBt_.resize(0);
    am_BCinv_b_.resize(0); 
    CinvBt_x_.resize(0);

    std::cout << "Reset bundle adjustment solver.\n";
};
void SparseBundleAdjustmentSolver::setParameterVectorFromPosesPoints(){
    // 1) Pose part
    for(int j_opt = 0; j_opt < N_opt_; ++j_opt){
        PoseSE3Tangent xi_jw;
        geometry::SE3Log_f(ba_params_->getOptPose(j_opt),xi_jw);
        params_poses_[j_opt] = xi_jw;
    }

    // 2) Point part
    for(int i = 0; i < M_; ++i) 
        params_points_[i] = ba_params_->getOptPoint(i);
};

void SparseBundleAdjustmentSolver::getPosesPointsFromParameterVector(){
    // Generate parameters
    // xi part 0~5, 6~11, ... 
    for(int j_opt = 0; j_opt < N_opt_; ++j_opt){
        PoseSE3 Tjw;
        geometry::se3Exp_f(params_poses_[j_opt], Tjw);
        ba_params_->updateOptPose(j_opt,Tjw);
    }
    // point part
    for(int i = 0; i < M_; ++i)
        ba_params_->updateOptPoint(i, params_points_[i]);
};

void SparseBundleAdjustmentSolver::zeroizeStorageMatrices(){
    // std::cout << "in zeroize \n";
    for(int j = 0; j < N_opt_; ++j){
        A_[j].setZero();
        a_[j].setZero();
        x_[j].setZero();
        BCinv_b_[j].setZero();
        am_BCinv_b_[j].setZero();

        for(int i = 0; i < M_; ++i){
            B_[j][i].setZero();
            Bt_[i][j].setZero();
            BCinv_[j][i].setZero();
            CinvBt_[i][j].setZero();
        }
        for(int k = 0; k < N_opt_; ++k){
            BCinvBt_[j][k].setZero();
            Am_BCinvBt_[j][k].setZero();
        }
    }
    for(int i = 0; i < M_; ++i){
        C_[i].setZero();
        Cinv_[i].setZero();
        b_[i].setZero();
        y_[i].setZero();
        Bt_x_[i].setZero();
        Cinv_b_[i].setZero();
        CinvBt_x_[i].setZero();
    }
};    
