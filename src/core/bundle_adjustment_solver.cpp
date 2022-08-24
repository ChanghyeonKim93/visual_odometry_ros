#include "core/bundle_adjustment_solver.h"
/*
    <Problem we want to solve>
        H*delta_theta = J.transpose()*r;

    where
    - Hessian  
            H = [A_, B_;
                    Bt_, C_];

    - Jacobian multiplied by residual vector 
            J.transpose()*r = [a;b];

    - Update parameters
            delta_theta = [x;y];
*/
// A sparse solver for a feature-based Bundle adjustment problem.
BundleAdjustmentSolver::BundleAdjustmentSolver() {
    cam_ = nullptr;

    A_.reserve(500); // reserve expected # of optimizable poses (N_opt)
    B_.reserve(500); // 
    Bt_.reserve(1000000);
    C_.reserve(1000000); // reserve expected # of optimizable landmarks (M)
};

// Set Huber threshold
void BundleAdjustmentSolver::setHuberThreshold(float thres_huber){
    THRES_HUBER_ = thres_huber;
};

// Set camera.
void BundleAdjustmentSolver::setCamera(const std::shared_ptr<Camera>& cam){
    cam_ = cam;
};

// Set problem sizes and resize the storages.
void BundleAdjustmentSolver::setProblemSize(int N, int N_opt, int M, int n_obs){ 
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

// Set Input Values.
void BundleAdjustmentSolver::setInitialValues(
    const std::map<FramePtr,PoseSE3>& Tjw_map,
    const LandmarkBAVec& lms_ba,
    const std::map<FramePtr,int>& kfmap_optimize)
{
    std::copy(Tjw_map.begin(), Tjw_map.end(), std::inserter(Tjw_map_, Tjw_map_.begin()));// Keyframes - Poses map
    std::copy(kfmap_optimize.begin(), kfmap_optimize.end(), std::inserter(kfmap_optimize_, kfmap_optimize_.begin())); // Keyframes to be opt. - Indexes map.
    
    kfvec_optimize_.resize(kfmap_optimize_.size()); // Indexes to keyframes to be opt. (mutually directing)
    for(auto kf : kfmap_optimize_) 
        kfvec_optimize_[kf.second] = kf.first;

    lms_ba_.resize(lms_ba.size()); // Landmarks
    std::copy(lms_ba.begin(),lms_ba.end(), lms_ba_.begin());

    if(M_ != lms_ba.size()) throw std::runtime_error("In BundleAdjustmentSolver, 'M_ != lms_ba.size()'.");
    if(N_ != Tjw_map_.size()) throw std::runtime_error("In BundleAdjustmentSolver, 'N_ != Tjw_map_.size()'.");
};

// Solve the BA for fixed number of iterations
void BundleAdjustmentSolver::solveForFiniteIterations(int MAX_ITER){

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
            const Point&       Xi  = lms_ba_[i].X; 
            const FramePtrVec& kfs = lms_ba_[i].kfs_seen;
            const PixelVec&    pts = lms_ba_[i].pts_on_kfs;

            for(int jj = 0; jj < kfs.size(); ++jj){
                // For j-th keyframe
                const Pixel&   pij = pts[jj];
                const FramePtr& kf = kfs[jj];

                // 0) check whether it is optimizable keyframe
                bool is_optimizable_keyframe = false;
                int j = -1;
                if(kfmap_optimize_.find(kf) != kfmap_optimize_.end() ) { // this is a opt. keyframe.
                    is_optimizable_keyframe = true;
                    j = kfmap_optimize_[kf]; // j_opt < N_opt_
                }

                // Current poses
                const PoseSE3& T_jw = Tjw_map_[kf];
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

            // For i-th landmark, fill other storages
            if(abs(C_[i].determinant()) < 0.0001) {
                // std::cout << i <<"-th point C[i] det: " <<C_[i].determinant() << std::endl;
                Cinv_[i] = 0.1*Mat33::Identity();
                C_[i] = 100*Mat33::Identity();
            }
            else Cinv_[i] = C_[i].inverse(); // FILL STORAGE (3-1)
            
            Cinv_b_[i] = Cinv_[i]*b_[i];  // FILL STORAGE (10)
            for(int jj = 0; jj < kfs.size(); ++jj){
                // For j-th keyframe
                const FramePtr& kf = kfs[jj];

                // 0) check whether it is optimizable keyframe
                bool is_optimizable_keyframe_j = false;
                int j = -1;
                if(kfmap_optimize_.find(kf) != kfmap_optimize_.end() ) { // this is a opt. keyframe.
                    is_optimizable_keyframe_j = true;
                    j = kfmap_optimize_[kf]; // j_opt < N_opt_

                    BCinv_[j][i]  = B_[j][i]*Cinv_[i];  // FILL STORAGE (6)
                    CinvBt_[i][j] = BCinv_[j][i].transpose(); // FILL STORAGE (11)
                    BCinv_b_[j]  += BCinv_[j][i]*b_[i];  // FILL STORAGE (9)
                }

                for(int kk = jj; kk < kfs.size(); ++kk){
                    // For k-th keyframe
                    const FramePtr& kf2 = kfs[kk];
                    bool is_optimizable_keyframe_k = false;
                    int k = -1;
                    if(kfmap_optimize_.find(kf2) != kfmap_optimize_.end() ) { // this is a opt. keyframe.
                        is_optimizable_keyframe_k = true;
                        k = kfmap_optimize_[kf2]; // j_opt < N_opt_
                    }

                    if(is_optimizable_keyframe_j && is_optimizable_keyframe_k){
                        Mat66 BCinvBt_tmp = BCinv_[j][i]*Bt_[i][k];
                        BCinvBt_[j][k] += BCinvBt_tmp;  // FILL STORAGE (7)
                    }
                }
            } // END jj
        } // END i

        for(int j = 0; j < N_opt_; ++j)
            for(int k = j; k < N_opt_; ++k)
                BCinvBt_[k][j] = BCinvBt_[j][k];
            
        for(int j = 0; j < N_opt_; ++j){
            for(int k = 0; k < N_opt_; ++k){
                if(j != k) Am_BCinvBt_[j][k] =       - BCinvBt_[j][k];
                else       Am_BCinvBt_[j][k] = A_[j] - BCinvBt_[j][k];
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
        
        for(int j = 0; j < N_opt_; ++j){
            x_[j] = x_mat.block<6,1>(6*j,0);
        }
        for(int i = 0; i < M_; ++i){
            const FramePtrVec& kfs = lms_ba_[i].kfs_seen;
            for(int jj = 0; jj < kfs.size(); ++jj){
                const FramePtr& kf = kfs[jj];
                int j = -1;
                if(kfmap_optimize_.find(kf) != kfmap_optimize_.end() ) { // this is a opt. keyframe.
                    j = kfmap_optimize_[kf]; // j_opt < N_opt_
                    CinvBt_x_[i] += CinvBt_[i][j]*x_[j];
                }
            }
            y_[i] = Cinv_b_[i] - CinvBt_x_[i];
        }
                
        // Update step
        for(int j = 0; j < N_opt_; ++j)
            params_poses_[j] += 0.3*x_[j];
        for(int i = 0; i < M_; ++i){
            if(abs(C_[i].determinant()) >= 0.0001) {
                params_points_[i] += 0.5*y_[i];
            }
        }

        std::cout << iter << "-th iter, error : " << 0.5f*err/(float)n_obs_ << "\n";
    } // END iter

    // Finally, update parameters
    for(int j = 0; j < N_opt_; ++j){
        PoseSE3 T_jw;
        geometry::se3Exp_f(params_poses_[j], T_jw);
        
        PoseSE3 T_jw_original = kfvec_optimize_[j]->getPoseInv();

        // std::cout << j << "-th pose: \n" << T_jw_original.inverse()*T_jw << std::endl;

        kfvec_optimize_[j]->setPose(T_jw.inverse());
        Tjw_map_[kfvec_optimize_[j]] = T_jw;
    }
    for(int i = 0; i < M_; ++i)
        lms_ba_[i].lm->set3DPoint(lms_ba_[i].X);

    // Finish
    // std::cout << "======= Local Bundle adjustment - sucess:" << (flag_success ? "SUCCESS" : "FAILED") << "=======\n";

};

// Reset local BA solver.
void BundleAdjustmentSolver::reset(){
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

    Tjw_map_.clear();
    kfmap_optimize_.clear();
    lms_ba_.resize(0);

    std::cout << "Reset bundle adjustment solver.\n";
};
void BundleAdjustmentSolver::setParameterVectorFromPosesPoints(){
    if(kfmap_optimize_.empty()) throw std::runtime_error("kfmap_optimize_.empty() == true.");        // initialize optimization parameter vector.
    // 1) Pose part
    for(auto kf : kfmap_optimize_){
        const FramePtr& f = kf.first;
        int j = kf.second;

        PoseSE3Tangent xi_jw;
        geometry::SE3Log_f(Tjw_map_[f],xi_jw);
        params_poses_[j] = xi_jw;
    }
    // 2) Point part
    for(int i = 0; i < M_; ++i) 
        params_points_[i] = lms_ba_[i].X;
};

void BundleAdjustmentSolver::getPosesPointsFromParameterVector(){
    // Generate parameters
    // xi part 0~5, 6~11, ... 
    int idx = 0;
    for(int j = 0; j < N_opt_; ++j){
        PoseSE3 T_jw;
        geometry::se3Exp_f(params_poses_[j], T_jw);
        Tjw_map_[kfvec_optimize_[j]] = T_jw;
    }
    // point part
    for(int i = 0; i < M_; ++i) 
        lms_ba_[i].X = params_points_[i];
};

void BundleAdjustmentSolver::zeroizeStorageMatrices(){
    // std::cout << "in zeroize \n";
    for(int j = 0; j < N_opt_; ++j){
        A_[j].setZero();
        a_[j].setZero();
        x_[j].setZero();
        // params_poses_[j].setZero();
        BCinv_b_[j].setZero();
        am_BCinv_b_[j].setZero();

        for(int i = 0; i < M_; ++i){
            B_[j][i].setZero();
            Bt_[i][j].setZero();
            BCinv_[j][i].setZero();
            CinvBt_[i][j].setZero();
        }
        for(int i = 0; i < N_opt_;++i){
            BCinvBt_[j][i].setZero();
            Am_BCinvBt_[j][i].setZero();
        }
    }
    for(int i = 0; i < M_; ++i){
        C_[i].setZero();
        Cinv_[i].setZero();
        b_[i].setZero();
        y_[i].setZero();
        // params_points_[i].setZero();
        Bt_x_[i].setZero();
        Cinv_b_[i].setZero();
        CinvBt_x_[i].setZero();
    }
};    