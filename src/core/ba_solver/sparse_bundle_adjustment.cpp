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



SparseBundleAdjustmentSolver::SparseBundleAdjustmentSolver(bool is_stereo) 
: N_(0), N_opt_(0), M_(0), n_obs_(0), THRES_EPS_(0), THRES_HUBER_(0),
n_cams_(0), is_stereo_mode_(is_stereo)
{
    cams_.resize(0);
    // this->cam_       = nullptr;
    this->ba_params_ = nullptr;

    A_.reserve(500); // reserve expected # of optimizable poses (N_opt)
    B_.reserve(500); // 
    Bt_.reserve(5000);
    C_.reserve(5000); // reserve expected # of optimizable landmarks (M)

    if(is_stereo_mode_)
        std::cout << "SparseBundleAdjustmentSolver() - set to 'stereo' mode.\n";
    else 
        std::cout << "SparseBundleAdjustmentSolver() - set to 'monocular' mode.\n";
};

void SparseBundleAdjustmentSolver::setBAParameters(const std::shared_ptr<SparseBAParameters>& ba_params)
{
    if(is_stereo_mode_){
        if( !ba_params->isStereoMode() )
            throw std::runtime_error("In SparseBundleAdjustmentSolver::setBAParameters(), 'ba_params' is not in stereo mode while 'is_stereo' of this module is set to 'true'.");
    }

    this->ba_params_ = ba_params;
    
    this->N_     = this->ba_params_->getNumOfAllFrames();
    this->N_opt_ = this->ba_params_->getNumOfOptimizeFrames();
    this->M_     = this->ba_params_->getNumOfOptimizeLandmarks();
    this->n_obs_ = this->ba_params_->getNumOfObservations();

    this->setProblemSize(N_, N_opt_, M_, n_obs_);
};

void SparseBundleAdjustmentSolver::setHuberThreshold(_BA_Numeric thres_huber){
    THRES_HUBER_ = thres_huber;
};

void SparseBundleAdjustmentSolver::setCamera(const std::shared_ptr<Camera>& cam){
     if( !is_stereo_mode_ ){
        cams_.resize(1);
        cams_[0] = cam;
        n_cams_ = 1;
        std::cout << "Sparse BA solver is in 'monocular mode'\n";   
    }
    else
        throw std::runtime_error("In 'SparseBundleAdjustmentSolver::setCamera()': Before call this function, 'is_stereo' should be set to 'false'.");
};

void SparseBundleAdjustmentSolver::setStereoCameras(const std::shared_ptr<Camera>& cam0, const std::shared_ptr<Camera>& cam1)
{
    if( is_stereo_mode_ ){
        cams_.resize(2);
        cams_[0] = cam0;
        cams_[1] = cam1;
        n_cams_ = 2;
        
        std::cout << "Sparse BA solver is in 'stereo mode'\n";
    }
    else
        throw std::runtime_error("In 'SparseBundleAdjustmentSolver::setStereoCameras()': Before call this function, 'is_stereo' should be set to 'true'.");
};


void SparseBundleAdjustmentSolver::setProblemSize(int N, int N_opt, int M, int n_obs){ 
    // Set sizes
    this->N_     = N; // # of total keyframes (including fixed frames)
    this->N_opt_ = N_opt; // # of optimizable keyframes
    this->M_     = M; // # of landmarks to be optimized.
    this->n_obs_ = n_obs;

    // Resize storages.
    A_.resize(N_opt_, _BA_Mat66::Zero()); 
    
    B_.resize(N_opt_);
    for(_BA_Index j = 0; j < N_opt_; ++j) 
        B_[j].resize(M_, _BA_Mat63::Zero());   // 6x3, N_opt X M blocks
    
    Bt_.resize(M_);    
    for(_BA_Index i = 0; i < M_; ++i) 
        Bt_[i].resize(N_opt_, _BA_Mat36::Zero());   // 3x6, N_opt X M blocks
    
    C_.resize(M_, _BA_Mat33::Zero());

    a_.resize(N_opt_, _BA_Vec6::Zero()); // 6x1, N_opt blocks
    x_.resize(N_opt_, _BA_Vec6::Zero()); // 6x1, N_opt blocks
    params_poses_.resize(N_opt_, _BA_PoseSE3::Identity()); // 4x4, N_opt blocks

    b_.resize(M_, _BA_Vec3::Zero());     // 3x1, M blocks
    y_.resize(M_, _BA_Vec3::Zero());     // 3x1, M blocks
    params_points_.resize(M_, _BA_Vec3::Zero());    // 3x1, M blocks

    Cinv_.resize(M_, _BA_Mat33::Zero()); // 3x3, M diagonal blocks 

    BCinv_.resize(N_opt_); 
    for(_BA_Index j = 0; j < N_opt_; ++j) 
        BCinv_[j].resize(M_, _BA_Mat63::Zero());   // 6x3, N_opt X M blocks
    
    CinvBt_.resize(M_); 
    for(_BA_Index i = 0 ; i < M_; ++i) 
        CinvBt_[i].resize(N_opt_, _BA_Mat36::Zero());
    
    BCinvBt_.resize(N_opt_); 
    for(_BA_Index j = 0; j < N_opt_; ++j) 
        BCinvBt_[j].resize(N_opt_, _BA_Mat66::Zero());   // 6x6, N_opt X N_opt blocks
    
    BCinv_b_.resize(N_opt_); // 6x1, N_opt x 1 blocks
    am_BCinv_b_.resize(N_opt_); // 6x1, N_opt x 1 blocks

    Am_BCinvBt_.resize(N_opt_); 
    for(_BA_Index j = 0; j < N_opt_; ++j) 
        Am_BCinvBt_[j].resize(N_opt_, _BA_Mat66::Zero());   // 6x6, N_opt X N_opt blocks
    
    Cinv_b_.resize(M_, _BA_Vec3::Zero());
    Bt_x_.resize(M_, _BA_Vec3::Zero());        // 3x1, M x 1 blocks
    CinvBt_x_.resize(M_, _BA_Vec3::Zero());


    // Dynamic matrices
    Am_BCinvBt_mat_.resize(6*N_opt_, 6*N_opt_);
    am_BCinv_b_mat_.resize(6*N_opt_, 1);
    x_mat_.resize(6*N_opt_, 1);

    Am_BCinvBt_mat_.setZero();
    am_BCinv_b_mat_.setZero();
    x_mat_.setZero();
};

bool SparseBundleAdjustmentSolver::solveForFiniteIterations(int MAX_ITER)
{
    bool flag_success    = true;
    bool flag_nan_pass   = true;
    bool flag_error_pass = true;

    std::cout << "LBA    : start to solve iteratively...\n";
    
    _BA_Numeric THRES_SUCCESS_AVG_ERROR = 1.0;

    _BA_Numeric MAX_LAMBDA = 20.0;
    _BA_Numeric MIN_LAMBDA = 1e-6;

    // For stereo extrinsics
    _BA_Numeric fx_r, fy_r, cx_r, cy_r;
    _BA_PoseSE3 T_lr, T_rl;
    _BA_Rot3 R_rl, R_lr;
    _BA_Pos3 t_rl, t_lr;
    if(is_stereo_mode_)
    { 
        fx_r = cams_[1]->fx(); fy_r = cams_[1]->fy();
        cx_r = cams_[1]->cx(); cy_r = cams_[1]->cy();

        T_lr = ba_params_->getStereoPose(); // left to right pose
        T_rl = geometry::inverseSE3(T_lr);
        R_lr = T_lr.block<3,3>(0,0); t_lr = T_lr.block<3,1>(0,3);
        R_rl = T_rl.block<3,3>(0,0); t_rl = T_rl.block<3,1>(0,3);
    }

    // Set the Parameter Vector.
    this->setParameterVectorFromPosesPoints();

    // Initialize parameters
    std::vector<_BA_Numeric> r_prev(n_obs_, 0.0f);
    _BA_Numeric err_prev = 1e10f;
    _BA_Numeric lambda   = 0.00001;
    std::cout << "start iteration\n";
    for(_BA_Index iter = 0; iter < MAX_ITER; ++iter)
    {
        // Reset A, B, Bt, C, Cinv, a, b, x, y...
        this->zeroizeStorageMatrices();

        // Iteratively solve. (Levenberg-Marquardt algorithm)
        _BA_Index   cnt = 0;
        _BA_Numeric err = 0.0f;
        
        // For i-th landmark
        for(_BA_Index i = 0; i < M_; ++i)
        {
            const LandmarkBA&   lmba = ba_params_->getLandmarkBA(i);

            const _BA_Point&    Xi  = lmba.X;
            const FramePtrVec&  kfs = lmba.kfs_seen;
            const _BA_PixelVec& pts = lmba.pts_on_kfs;

            // For j-th keyframe
            for(_BA_Index jj = 0; jj < kfs.size(); ++jj) 
            {
                const _BA_Pixel& pij = pts.at(jj);
                FramePtr          kf = kfs.at(jj);

                if( kf->isRightImage() )
                {
                    // In case of stereo second image, 
                    // get related major image pointer
                    kf = kf->getLeftFramePtr();

                    // Current major pose
                    const _BA_PoseSE3& T_jw = ba_params_->getPose(kf);
                    const _BA_Rot3& R_jw = T_jw.block<3,3>(0,0);
                    const _BA_Pos3& t_jw = T_jw.block<3,1>(0,3);
                    _BA_Rot3 R_rl_R_jw = R_rl*R_jw;

                    _BA_Point Xij = R_jw*Xi + t_jw; // transform a 3D point.
                    _BA_Point Xijr = R_rl*Xij + t_rl;
                    

                    
                    // 0) check whether it is optimizable keyframe
                    bool is_opt_keyframe = false;
                    _BA_Index j = -1;
                    if(ba_params_->isOptFrame(kf))
                    {
                        is_opt_keyframe = true;
                        j = ba_params_->getOptPoseIndex(kf);
                    }

                
                    // 1) Qij and Rij calculation.
                    const _BA_Numeric& xjr = Xijr(0), yjr = Xijr(1), zjr = Xijr(2);
                    _BA_Numeric invzr = 1.0f/zjr; _BA_Numeric invzr2 = invzr*invzr;
                    
                    _BA_Numeric fxinvz      = fx_r*invzr;   _BA_Numeric fyinvz    = fy_r*invzr;
                    _BA_Numeric xinvz       = xjr*invzr;    _BA_Numeric yinvz     = yjr*invzr;
                    _BA_Numeric fx_xinvz2   = fxinvz*xinvz; _BA_Numeric fy_yinvz2 = fyinvz*yinvz;
                    _BA_Numeric xinvz_yinvz = xinvz*yinvz;

                    // 1) residual calculation
                    _BA_Pixel ptw;
                    ptw << fx_r*xinvz + cx_r, fy_r*yinvz + cy_r;
                    _BA_Vec2 rij;
                    rij = ptw - pij;

                    // 2) HUBER weight calculation (Manhattan distance)
                    _BA_Numeric absrxry = abs(rij(0)) + abs(rij(1));
                    
                    r_prev[cnt] = absrxry;
                    
                    _BA_Numeric weight = 1.0;
                    bool flag_weight = false;
                    if(absrxry > THRES_HUBER_) {
                        weight = (THRES_HUBER_/absrxry);
                        flag_weight = true;
                    }

                    // 3) Rij calculation (Jacobian w.r.t. Xi)
                    _BA_Mat23 Rij;
                    const _BA_Numeric& r11 = R_rl_R_jw(0,0), r12 = R_rl_R_jw(0,1), r13 = R_rl_R_jw(0,2);
                    const _BA_Numeric& r21 = R_rl_R_jw(1,0), r22 = R_rl_R_jw(1,1), r23 = R_rl_R_jw(1,2);
                    const _BA_Numeric& r31 = R_rl_R_jw(2,0), r32 = R_rl_R_jw(2,1), r33 = R_rl_R_jw(2,2);
                    Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                           fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33;
                    
                    _BA_Mat32 Rij_t = Rij.transpose();

                    _BA_Mat33 Rij_t_Rij;
                    _BA_Vec3  Rij_t_rij;

                    this->calc_Rij_t_Rij_weight(weight, Rij, Rij_t_Rij);
                    Rij_t_rij = weight * (Rij_t*rij);

                    
                    C_[i].noalias() +=  Rij_t_Rij; // FILL STORAGE (3)
                    b_[i].noalias() += -Rij_t_rij; // FILL STORAGE (5)

                    if(is_opt_keyframe) 
                    {
                        _BA_Mat23 dp_dX;
                        dp_dX << fxinvz, 0, -fx_xinvz2, 0, fyinvz, -fy_yinvz2;
                        _BA_Mat33 Xij_skew; Xij_skew << 0, -Xij(2), Xij(1), Xij(2), 0, -Xij(0), -Xij(1), Xij(0), 0;

                        // Optimizable keyframe.
                        _BA_Mat26 Qij;
                        Qij << dp_dX*R_rl, -dp_dX*R_rl*Xij_skew;
                            
                        _BA_Mat62 Qij_t = Qij.transpose();

                        _BA_Mat66 Qij_t_Qij; Qij_t_Qij.setZero();
                        _BA_Mat63 Qij_t_Rij; Qij_t_Rij.setZero();
                        _BA_Vec6 Qij_t_rij; Qij_t_rij.setZero();
                                            
                        this->calc_Qij_t_Qij_weight(weight, Qij, Qij_t_Qij);
                        Qij_t_Rij = weight * (Qij_t*Rij);
                        Qij_t_rij = weight * (Qij_t*rij);


                        A_[j].noalias() +=  Qij_t_Qij; // FILL STORAGE (1)
                        B_[j][i]         =  Qij_t_Rij; // FILL STORAGE (2)
                        Bt_[i][j]        =  Qij_t_Rij.transpose().eval(); // FILL STORAGE (2-1)
                        a_[j].noalias() += -Qij_t_rij; // FILL STORAGE (4)

                        if(std::isnan(Qij_t_Qij.norm()) || std::isnan(Qij_t_Rij.norm()) || std::isnan(Qij_t_rij.norm()) )
                            throw std::runtime_error("In   LBA, pose becomes nan!");

                    } // END is_opt_frame

                   _BA_Numeric err_tmp = rij.transpose()*rij;
                    err += err_tmp;

                    ++cnt;
                } // END  if( kf->isRightImage() == true )
                else
                {
                    // 0) check whether it is optimizable keyframe
                    _BA_Index j = -1;
                    bool is_opt_frame = ba_params_->isOptFrame(kf);
                    if(is_opt_frame)
                        j = ba_params_->getOptPoseIndex(kf);

                    // Get current camera parameters
                    const _BA_Numeric& fx = cams_[0]->fx(), fy = cams_[0]->fy() ,cx = cams_[0]->cx(), cy = cams_[0]->cy();

                    // Current poses
                    const _BA_PoseSE3& T_jw = ba_params_->getPose(kf);
                    const _BA_Rot3& R_jw = T_jw.block<3,3>(0,0);
                    const _BA_Pos3& t_jw = T_jw.block<3,1>(0,3);
                    
                    _BA_Point Xij = R_jw*Xi + t_jw; // transform a 3D point.
                
                    // 1) Qij and Rij calculation.
                    const _BA_Numeric& xj = Xij(0), yj = Xij(1), zj = Xij(2);
                    _BA_Numeric invz = 1.0f/zj; _BA_Numeric invz2 = invz*invz;
                    
                    _BA_Numeric fxinvz      = fx*invz;      _BA_Numeric fyinvz      = fy*invz;
                    _BA_Numeric xinvz       = xj*invz;      _BA_Numeric yinvz       = yj*invz;
                    _BA_Numeric fx_xinvz2   = fxinvz*xinvz; _BA_Numeric fy_yinvz2   = fyinvz*yinvz;
                    _BA_Numeric xinvz_yinvz = xinvz*yinvz;

                    // 1) residual calculation
                    _BA_Pixel ptw;
                    ptw << fx*xinvz + cx, fy*yinvz + cy;
                    _BA_Vec2 rij;
                    rij = ptw - pij;

                    // 2) HUBER weight calculation (Manhattan distance)
                    _BA_Numeric absrxry = abs(rij(0)) + abs(rij(1));
                    
                    r_prev[cnt] = absrxry;
                    
                    _BA_Numeric weight = 1.0;
                    bool flag_weight = false;
                    if(absrxry > THRES_HUBER_) {
                        weight = (THRES_HUBER_/absrxry);
                        flag_weight = true;
                    }

                    // 3) Rij calculation (Jacobian w.r.t. Xi)
                    _BA_Mat23 Rij;
                    const _BA_Numeric& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2);
                    const _BA_Numeric& r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2);
                    const _BA_Numeric& r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
                    Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                        fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33;
                    
                    _BA_Mat32 Rij_t = Rij.transpose();

                    _BA_Mat33 Rij_t_Rij;
                    _BA_Vec3  Rij_t_rij;

                    this->calc_Rij_t_Rij_weight(weight, Rij, Rij_t_Rij);
                    Rij_t_rij = weight * (Rij_t*rij);

                    
                    C_[i].noalias() +=  Rij_t_Rij; // FILL STORAGE (3)
                    b_[i].noalias() += -Rij_t_rij; // FILL STORAGE (5)

                    if(is_opt_frame) 
                    {
                        // Optimizable keyframe.
                        _BA_Mat26 Qij;
                        Qij << fxinvz,0,-fx_xinvz2,-fx*xinvz_yinvz,fx*(1.0+xinvz*xinvz), -fx*yinvz,
                            0,fyinvz,-fy_yinvz2,-fy*(1.0+yinvz*yinvz),fy*xinvz_yinvz,  fy*xinvz;

                        _BA_Mat62 Qij_t = Qij.transpose();

                        _BA_Mat66 Qij_t_Qij; Qij_t_Qij.setZero();
                        _BA_Mat63 Qij_t_Rij; Qij_t_Rij.setZero();
                        _BA_Vec6 Qij_t_rij; Qij_t_rij.setZero();
                                            
                        this->calc_Qij_t_Qij_weight(weight, Qij, Qij_t_Qij);
                        Qij_t_Rij = weight * (Qij_t*Rij);
                        Qij_t_rij = weight * (Qij_t*rij);


                        A_[j].noalias() +=  Qij_t_Qij; // FILL STORAGE (1)
                        B_[j][i]         =  Qij_t_Rij; // FILL STORAGE (2)
                        Bt_[i][j]        =  Qij_t_Rij.transpose().eval(); // FILL STORAGE (2-1)
                        a_[j].noalias() += -Qij_t_rij; // FILL STORAGE (4)

                        if(std::isnan(Qij_t_Qij.norm()) || std::isnan(Qij_t_Rij.norm()) || std::isnan(Qij_t_rij.norm()) )
                            throw std::runtime_error("In   LBA, pose becomes nan!");

                    } // END is_opt_frame

                    _BA_Numeric err_tmp = rij.transpose()*rij;
                    err += err_tmp;

                    ++cnt;
                } // END  if( kf->isRightImage() == FALSE ) // (left image)
                
            } // END jj of i-th point
        } // END i-th point
  
        // Solve sequentially.
        // 1) Damping 'A_' diagonal
        for(_BA_Index j = 0; j < N_opt_; ++j)
        {
            _BA_Mat66& A_tmp = A_[j];
            A_tmp(0,0) += lambda*A_tmp(0,0);
            A_tmp(1,1) += lambda*A_tmp(1,1);
            A_tmp(2,2) += lambda*A_tmp(2,2);
            A_tmp(3,3) += lambda*A_tmp(3,3);
            A_tmp(4,4) += lambda*A_tmp(4,4);
            A_tmp(5,5) += lambda*A_tmp(5,5);
        }

        // 2) Damping 'C_' diagonal, and Calculate 'Cinv_' & 'Cinvb_'
        for(_BA_Index i = 0; i < M_; ++i)
        {
            _BA_Mat33& C_tmp = C_[i];
            // Add damping coefficient for i-th point
            C_tmp(0,0) += lambda*C_tmp(0,0);
            C_tmp(1,1) += lambda*C_tmp(1,1);
            C_tmp(2,2) += lambda*C_tmp(2,2);

            Cinv_[i] = C_[i].ldlt().solve(_BA_Mat33::Identity());
            Cinv_b_[i] = Cinv_[i]*b_[i];  // FILL STORAGE (10)
        }

        // 3) Calculate 'BCinv_', 'BCinvb_',' BCinvBt_'
        for(_BA_Index i = 0; i < M_; ++i)
        {
            const FramePtrVec&  kfs  = ba_params_->getLandmarkBA(i).kfs_seen;
            for(_BA_Index jj = 0; jj < kfs.size(); ++jj) 
            {
                // For j-th keyframe
                const FramePtr& kf = kfs[jj];

                if(kf->isRightImage())
                    continue; // skip the right image

                // 0) check whether it is optimizable keyframe
                bool is_optimizable_keyframe_j = false;
                _BA_Index j = -1;
                if(ba_params_->isOptFrame(kf))
                {
                    is_optimizable_keyframe_j = true;
                    j = ba_params_->getOptPoseIndex(kf);

                    BCinv_[j][i]  = B_[j][i]*Cinv_[i];  // FILL STORAGE (6)
                    CinvBt_[i][j] = BCinv_[j][i].transpose().eval(); // FILL STORAGE (11)
                    BCinv_b_[j].noalias() += BCinv_[j][i]*b_[i];  // FILL STORAGE (9)

                    for(_BA_Index kk = jj; kk < kfs.size(); ++kk)
                    {
                        // For k-th keyframe
                        const FramePtr& kf2 = kfs[kk];

                        if(kf2->isRightImage()) 
                            continue; // skip the second image

                        bool is_optimizable_keyframe_k = false;
                        _BA_Index k = -1;
                        if(ba_params_->isOptFrame(kf2))
                        {
                            is_optimizable_keyframe_k = true;
                            k = ba_params_->getOptPoseIndex(kf2);

                            BCinvBt_[j][k].noalias() += BCinv_[j][i]*Bt_[i][k];  // FILL STORAGE (7)
                        }
                    }
                }
            } // END jj of i-th point
        } // END i-th point

        for(_BA_Index j = 0; j < N_opt_; ++j)
            for(_BA_Index u = j; u < N_opt_; ++u)
                BCinvBt_[u][j] = BCinvBt_[j][u].transpose().eval();
            
        for(_BA_Index j = 0; j < N_opt_; ++j)
        {
            for(_BA_Index u = 0; u < N_opt_; ++u)
            {
                if(j == u) Am_BCinvBt_[j][u] = A_[j] - BCinvBt_[j][u];
                else       Am_BCinvBt_[j][u] =       - BCinvBt_[j][u];
            }
        }

        for(_BA_Index j = 0; j < N_opt_; ++j)
            am_BCinv_b_[j] = a_[j] - BCinv_b_[j];
        
        // Solve problem.
        // 1) solve x (w.r.t. pose)
        _BA_MatX& Am_BCinvBt_mat = Am_BCinvBt_mat_;
        _BA_MatX& am_BCinv_b_mat = am_BCinv_b_mat_;

        _BA_Index idx0 = 0;
        for(_BA_Index j = 0; j < N_opt_; ++j, idx0 += 6)
        {
            _BA_Index idx1 = 0;
            for(_BA_Index u = 0; u < N_opt_; ++u, idx1 += 6)
                Am_BCinvBt_mat.block(idx0,idx1,6,6) = Am_BCinvBt_[j][u];

            am_BCinv_b_mat.block(idx0,0,6,1) = am_BCinv_b_[j];
        }

        _BA_MatX& x_mat = x_mat_;
        x_mat = Am_BCinvBt_mat.ldlt().solve(am_BCinv_b_mat);
        idx0 = 0;
        for(_BA_Index j = 0; j < N_opt_; ++j, idx0 += 6)
            x_[j] = x_mat.block<6,1>(idx0,0);
        
        // 2) solve y (w.r.t. point)
        for(_BA_Index i = 0; i < M_; ++i) 
        {
            const LandmarkBA& lmba = ba_params_->getLandmarkBA(i);
            const FramePtrVec& kfs = lmba.kfs_seen;
            for(_BA_Index jj = 0; jj < kfs.size(); ++jj)
            {
                const FramePtr& kf = kfs[jj];

                if( kf->isRightImage() ) 
                    continue; // skip the second image.


                if(ba_params_->isOptFrame(kf))
                {
                    _BA_Index j = ba_params_->getOptPoseIndex(kf);
                    CinvBt_x_[i].noalias() += CinvBt_[i][j]*x_[j];
                }
            }
            y_[i] = Cinv_b_[i] - CinvBt_x_[i];
        }

        // NAN check
        for(_BA_Index i = 0; i < M_; ++i)
        {
            if(std::isnan(y_[i].norm()))
            {
                std::cout << i << "-th y_ is nan:\n";
                std::cout << "point: " << ba_params_->getLandmarkBA(i).X.transpose() <<std::endl;
                std::cout << "y_: " << y_[i].transpose() <<std::endl;
                std::cout << "C_[i]:\n" << C_[i] << std::endl;
                std::cout << "determinant: " << C_[i].determinant() <<std::endl;
                std::cout << "Cinv_[i]:\n" << Cinv_[i] << std::endl;
            }
        }
        for(_BA_Index j = 0; j < N_opt_; ++j)
        {
            if(std::isnan(x_[j].norm()))
            {
                std::cout << j << "-th x_ is nan:\n";
                std::cout << x_[j].transpose() << std::endl;
                std::cout << A_[j] << std::endl;
            }
        }
                
        // Update step
        for(_BA_Index j = 0; j < N_opt_; ++j)
        {
            _BA_PoseSE3 dT;
            geometry::se3Exp(x_[j], dT);
            params_poses_[j].noalias() = (dT*params_poses_[j]);
        }

        for(_BA_Index i = 0; i < M_; ++i)
            params_points_[i].noalias() += y_[i];

        this->getPosesPointsFromParameterVector();


        // Error calculation
        // Pixel error
        _BA_Numeric average_error = std::sqrt(err/(_BA_Numeric)n_obs_);
            
        std::cout << "LBA/ " << iter << "-th itr, err: " << average_error << " [px]\n";

        // Check extraordinary cases.
        flag_nan_pass   = std::isnan(err) ? false : true;
        flag_error_pass = (average_error <= THRES_SUCCESS_AVG_ERROR) ? true : false;
        flag_success    = flag_nan_pass && flag_error_pass;

        if( !flag_nan_pass ) {
            for(_BA_Index j = 0; j < N_opt_; ++j)
                std::cout << j << "-th A nan:\n" << A_[j] << std::endl;
            for(_BA_Index i = 0; i < M_; ++i)
                std::cout << i << "-th Cinv nan:\n" << Cinv_[i] << std::endl;
            for(_BA_Index j = 0; j < N_opt_; ++j)
                std::cout << j << "-th x nan:\n" << x_[j] << std::endl;
            for(_BA_Index i = 0; i < M_; ++i)
                std::cout << i << "-th y nan:\n" << y_[i] << std::endl;
            throw std::runtime_error("nan ......n.n,dgfmksermfoiaejrof");
        }
    } // END iter-th iteration



    // Finally, update parameters
    if(flag_nan_pass)
    {
        bool flag_large_update = false;

        // 1. Update optimizable 'major (left)' poses.
        for(_BA_Index j_opt = 0; j_opt < N_opt_; ++j_opt)
        {
            const FramePtr& kf = ba_params_->getOptFramePtr(j_opt);

            const PoseSE3& Twj_original_float = kf->getPose();
            _BA_PoseSE3 Twj_original;
            Twj_original << Twj_original_float(0,0), Twj_original_float(0,1), Twj_original_float(0,2), Twj_original_float(0,3), 
                            Twj_original_float(1,0), Twj_original_float(1,1), Twj_original_float(1,2), Twj_original_float(1,3), 
                            Twj_original_float(2,0), Twj_original_float(2,1), Twj_original_float(2,2), Twj_original_float(2,3), 
                            0.0, 0.0, 0.0, 1.0; 
            
            _BA_PoseSE3 Tjw_update = ba_params_->getPose(kf);
            Tjw_update = ba_params_->recoverOriginalScalePose(Tjw_update);
            Tjw_update = ba_params_->changeInvPoseRefToWorld(Tjw_update);

            // std::cout << j_opt << "-th pose changes:\n" << kf->getPoseInv() << "\n-->\n" << Tjw_update << std::endl;
            _BA_PoseSE3 dT = Twj_original*Tjw_update;
            if(dT.block<3,1>(0,3).norm() > 50) 
                flag_large_update = true;

            PoseSE3 Tjw_update_float;
            Tjw_update_float << Tjw_update(0,0),Tjw_update(0,1),Tjw_update(0,2),Tjw_update(0,3),
                                Tjw_update(1,0),Tjw_update(1,1),Tjw_update(1,2),Tjw_update(1,3),
                                Tjw_update(2,0),Tjw_update(2,1),Tjw_update(2,2),Tjw_update(2,3),
                                0.0, 0.0, 0.0, 1.0; 
            kf->setPose(geometry::inverseSE3_f(Tjw_update_float));
        }
        
        if(is_stereo_mode_)
        {            
            _BA_PoseSE3 T_lr_scaled = ba_params_->getStereoPose();
            T_lr_scaled = ba_params_->recoverOriginalScalePose(T_lr_scaled);

            PoseSE3 T_lr_f = T_lr.cast<float>();

            // 2. Update the second (right) poses. 
            // The related poses are already updated above.
            for(const auto& f : ba_params_->getAllFrameset())
            {
                if(f->isRightImage())
                {
                    const FramePtr& f_left = f->getLeftFramePtr();
                    const PoseSE3& T_wj = f->getPoseInv();

                    f->setPose(T_wj*T_lr_f);
                }
            }
        }
        
        for(_BA_Index i = 0; i < M_; ++i)
        {
            const LandmarkBA& lmba = ba_params_->getLandmarkBA(i);
            const LandmarkPtr& lm = lmba.lm;
            _BA_Point X_updated = lmba.X;

            X_updated = ba_params_->recoverOriginalScalePoint(X_updated);
            X_updated = ba_params_->warpToWorld(X_updated);

            _BA_Point X_original;
            X_original << lm->get3DPoint()(0),lm->get3DPoint()(1),lm->get3DPoint()(2);
            // std::cout << i << "-th lm changes: " << X_original.transpose() << " --> " << X_updated.transpose() << std::endl;
            if((X_original-X_updated).norm() > 2)
            {
                // for(int jj = 0; jj < lmba.pts_on_kfs.size(); ++jj){
                    // std::cout << " " << lmba.pts_on_kfs[jj].transpose();
                // }
                // std::cout <<"\n";
            }
            
            Point X_update_float;
            X_update_float << X_updated(0),X_updated(1),X_updated(2);

            lm->set3DPoint(X_update_float);
            if(X_update_float.norm() <= 1000)
                lm->setBundled();
            else
                lm->setDead();
        }

        if(flag_large_update)
        {
            for(_BA_Index j_opt = 0; j_opt < N_opt_; ++j_opt)
            {
                std::cout << j_opt << "-th pose:\n"
                        << params_poses_[j_opt] << std::endl;
                std::cout << "rot det:" <<  params_poses_[j_opt].block<3,3>(0,0).determinant() << std::endl;

            }
            throw std::runtime_error("large update!");
        }
    }
    else
    {
        std::vector<int> cnt_seen(N_+1,0);
        for(_BA_Index j = 0; j < N_opt_; ++j)
        {
            std::cout << j << "-th Pose:\n";
            std::cout << ba_params_->getOptPose(j) << std::endl;
            std::cout << "rot det:" <<  ba_params_->getOptPose(j).block<3,3>(0,0).determinant() << std::endl;
        }
        
        for(_BA_Index i = 0; i < M_; ++i)
        {
            const LandmarkBA&   lmba = ba_params_->getLandmarkBA(i);
            const _BA_Point&    Xi   = lmba.X; 
            const FramePtrVec&  kfs  = lmba.kfs_seen;
            const _BA_PixelVec& pts  = lmba.pts_on_kfs;
            cnt_seen[kfs.size()]++;

            std::cout << i << "-th lm point: " << Xi.transpose() <<std::endl;
            for(_BA_Index j = 0; j < pts.size(); ++j)
            {
                std::cout << i <<"-th lm, " << j << "-th obs: " << pts[j].transpose() <<std::endl;
            }
        }

        std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
        std::cout << "************************* LOCAL BA FAILED!!!!! ****************************\n";
        throw std::runtime_error("Local BA NAN!\n");        
    }

    // Finish
    // std::cout << "======= Local Bundle adjustment - sucess:" << (flag_success ? "SUCCESS" : "FAILED") << "=======\n";

    return flag_success;
};

void SparseBundleAdjustmentSolver::reset()
{
    ba_params_ = nullptr;
    cams_.resize(0);

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
    // std::cout << "Reset bundle adjustment solver.\n";
};

void SparseBundleAdjustmentSolver::setParameterVectorFromPosesPoints()
{
    // 1) Pose part
    for(_BA_Index j_opt = 0; j_opt < N_opt_; ++j_opt)
    {
        _BA_PoseSE3 Tjw = ba_params_->getOptPose(j_opt);
        params_poses_[j_opt].noalias() =  Tjw;
    }

    // 2) Point part
    for(_BA_Index i = 0; i < M_; ++i) 
        params_points_[i].noalias() =  ba_params_->getOptPoint(i);
};

void SparseBundleAdjustmentSolver::getPosesPointsFromParameterVector()
{
    // Generate parameters
    // xi part 0~5, 6~11, ... 
    for(_BA_Index j_opt = 0; j_opt < N_opt_; ++j_opt)
    {
        if(std::isnan(params_poses_[j_opt].norm()))
        {
            std::cerr << "std::isnan params_poses !\n";
            std::cout << params_poses_[j_opt] << std::endl;
        }
            
        _BA_PoseSE3 Tjw = params_poses_[j_opt];
        ba_params_->updateOptPose(j_opt, Tjw);
        std::cout << j_opt << "-th trans: " << 10.0 * Tjw.block<3,1>(0,3).transpose() <<", rot det: " << Tjw.block<3,3>(0,0).determinant() << std::endl;
    }
    // point part
    for(_BA_Index i = 0; i < M_; ++i)
        ba_params_->updateOptPoint(i, params_points_[i]);
};

void SparseBundleAdjustmentSolver::zeroizeStorageMatrices()
{
    // std::cout << "in zeroize \n";
    for(_BA_Index j = 0; j < N_opt_; ++j)
    {
        A_[j].setZero();
        a_[j].setZero();
        x_[j].setZero();
        BCinv_b_[j].setZero();
        am_BCinv_b_[j].setZero();

        for(_BA_Index i = 0; i < M_; ++i)
        {
            B_[j][i].setZero();
            Bt_[i][j].setZero();
            BCinv_[j][i].setZero();
            CinvBt_[i][j].setZero();
        }
        for(_BA_Index k = 0; k < N_opt_; ++k)
        {
            BCinvBt_[j][k].setZero();
            Am_BCinvBt_[j][k].setZero();
        }
    }
    for(_BA_Index i = 0; i < M_; ++i)
    {
        C_[i].setZero();
        Cinv_[i].setZero();
        b_[i].setZero();
        y_[i].setZero();
        Bt_x_[i].setZero();
        Cinv_b_[i].setZero();
        CinvBt_x_[i].setZero();
    }

    // Dynamic matrices
    Am_BCinvBt_mat_.setZero();
    am_BCinv_b_mat_.setZero();
    x_mat_.setZero();

    // std::cout << "zeroize done\n";
};    



// For fast calculations for symmetric matrices
inline void SparseBundleAdjustmentSolver::calc_Rij_t_Rij(const _BA_Mat23& Rij, 
    _BA_Mat33& Rij_t_Rij)
{
    Rij_t_Rij.setZero();

    // Calculate upper triangle
    const _BA_Mat23& a = Rij;
    Rij_t_Rij(0,0) = (a(0,0)*a(0,0) + a(1,0)*a(1,0));
    Rij_t_Rij(0,1) = (a(0,0)*a(0,1) + a(1,0)*a(1,1));
    Rij_t_Rij(0,2) = (a(0,0)*a(0,2) + a(1,0)*a(1,2));
    
    Rij_t_Rij(1,1) = (a(0,1)*a(0,1) + a(1,1)*a(1,1));
    Rij_t_Rij(1,2) = (a(0,1)*a(0,2) + a(1,1)*a(1,2));
    
    Rij_t_Rij(2,2) = (a(0,2)*a(0,2) + a(1,2)*a(1,2));

    // Substitute symmetric elements
    Rij_t_Rij(1,0) = Rij_t_Rij(0,1);
    Rij_t_Rij(2,0) = Rij_t_Rij(0,2);

    Rij_t_Rij(2,1) = Rij_t_Rij(1,2);
};

inline void SparseBundleAdjustmentSolver::calc_Rij_t_Rij_weight(const _BA_Numeric weight, const _BA_Mat23& Rij,
    _BA_Mat33& Rij_t_Rij)
{
    Rij_t_Rij.setZero();

    // Calculate upper triangle
    const _BA_Mat23& a = Rij;
    Rij_t_Rij(0,0) = weight*(a(0,0)*a(0,0) + a(1,0)*a(1,0));
    Rij_t_Rij(0,1) = weight*(a(0,0)*a(0,1) + a(1,0)*a(1,1));
    Rij_t_Rij(0,2) = weight*(a(0,0)*a(0,2) + a(1,0)*a(1,2));
    
    Rij_t_Rij(1,1) = weight*(a(0,1)*a(0,1) + a(1,1)*a(1,1));
    Rij_t_Rij(1,2) = weight*(a(0,1)*a(0,2) + a(1,1)*a(1,2));
    
    Rij_t_Rij(2,2) = weight*(a(0,2)*a(0,2) + a(1,2)*a(1,2));

    // Substitute symmetric elements
    Rij_t_Rij(1,0) = Rij_t_Rij(0,1);
    Rij_t_Rij(2,0) = Rij_t_Rij(0,2);

    Rij_t_Rij(2,1) = Rij_t_Rij(1,2);
};

inline void SparseBundleAdjustmentSolver::calc_Qij_t_Qij(const _BA_Mat26& Qij,
    _BA_Mat66& Qij_t_Qij)
{
    Qij_t_Qij.setZero();

    // a(0,1) = 0;
    // a(1,0) = 0;

    // Calculate upper triangle
    const _BA_Mat26& a = Qij;
    Qij_t_Qij(0,0) = (a(0,0)*a(0,0)                );
    Qij_t_Qij(0,1) = (a(0,0)*a(0,1)                );
    Qij_t_Qij(0,2) = (a(0,0)*a(0,2)                );
    Qij_t_Qij(0,3) = (a(0,0)*a(0,3)                );
    Qij_t_Qij(0,4) = (a(0,0)*a(0,4)                );
    Qij_t_Qij(0,5) = (a(0,0)*a(0,5)                );
    
    Qij_t_Qij(1,1) = (                a(1,1)*a(1,1));
    Qij_t_Qij(1,2) = (                a(1,1)*a(1,2));
    Qij_t_Qij(1,3) = (                a(1,1)*a(1,3));
    Qij_t_Qij(1,4) = (                a(1,1)*a(1,4));
    Qij_t_Qij(1,5) = (                a(1,1)*a(1,5));
    
    Qij_t_Qij(2,2) = (a(0,2)*a(0,2) + a(1,2)*a(1,2));
    Qij_t_Qij(2,3) = (a(0,2)*a(0,3) + a(1,2)*a(1,3));
    Qij_t_Qij(2,4) = (a(0,2)*a(0,4) + a(1,2)*a(1,4));
    Qij_t_Qij(2,5) = (a(0,2)*a(0,5) + a(1,2)*a(1,5));

    Qij_t_Qij(3,3) = (a(0,3)*a(0,3) + a(1,3)*a(1,3));
    Qij_t_Qij(3,4) = (a(0,3)*a(0,4) + a(1,3)*a(1,4));
    Qij_t_Qij(3,5) = (a(0,3)*a(0,5) + a(1,3)*a(1,5));
    
    Qij_t_Qij(4,4) = (a(0,4)*a(0,4) + a(1,4)*a(1,4));
    Qij_t_Qij(4,5) = (a(0,4)*a(0,5) + a(1,4)*a(1,5));
    
    Qij_t_Qij(5,5) = (a(0,5)*a(0,5) + a(1,5)*a(1,5));

    // Qij_t_Qij(0,0) = (a(0,0)*a(0,0) + a(1,0)*a(1,0));
    // Qij_t_Qij(0,1) = (a(0,0)*a(0,1) + a(1,0)*a(1,1));
    // Qij_t_Qij(0,2) = (a(0,0)*a(0,2) + a(1,0)*a(1,2));
    // Qij_t_Qij(0,3) = (a(0,0)*a(0,3) + a(1,0)*a(1,3));
    // Qij_t_Qij(0,4) = (a(0,0)*a(0,4) + a(1,0)*a(1,4));
    // Qij_t_Qij(0,5) = (a(0,0)*a(0,5) + a(1,0)*a(1,5));
    
    // Qij_t_Qij(1,1) = (a(0,1)*a(0,1) + a(1,1)*a(1,1));
    // Qij_t_Qij(1,2) = (a(0,1)*a(0,2) + a(1,1)*a(1,2));
    // Qij_t_Qij(1,3) = (a(0,1)*a(0,3) + a(1,1)*a(1,3));
    // Qij_t_Qij(1,4) = (a(0,1)*a(0,4) + a(1,1)*a(1,4));
    // Qij_t_Qij(1,5) = (a(0,1)*a(0,5) + a(1,1)*a(1,5));
    
    // Qij_t_Qij(2,2) = (a(0,2)*a(0,2) + a(1,2)*a(1,2));
    // Qij_t_Qij(2,3) = (a(0,2)*a(0,3) + a(1,2)*a(1,3));
    // Qij_t_Qij(2,4) = (a(0,2)*a(0,4) + a(1,2)*a(1,4));
    // Qij_t_Qij(2,5) = (a(0,2)*a(0,5) + a(1,2)*a(1,5));

    // Qij_t_Qij(3,3) = (a(0,3)*a(0,3) + a(1,3)*a(1,3));
    // Qij_t_Qij(3,4) = (a(0,3)*a(0,4) + a(1,3)*a(1,4));
    // Qij_t_Qij(3,5) = (a(0,3)*a(0,5) + a(1,3)*a(1,5));
    
    // Qij_t_Qij(4,4) = (a(0,4)*a(0,4) + a(1,4)*a(1,4));
    // Qij_t_Qij(4,5) = (a(0,4)*a(0,5) + a(1,4)*a(1,5));
    
    // Qij_t_Qij(5,5) = (a(0,5)*a(0,5) + a(1,5)*a(1,5));
    

    // Substitute symmetric elements
    // Qij_t_Qij(1,0) = Qij_t_Qij(0,1);
    Qij_t_Qij(2,0) = Qij_t_Qij(0,2);
    Qij_t_Qij(3,0) = Qij_t_Qij(0,3);
    Qij_t_Qij(4,0) = Qij_t_Qij(0,4);
    Qij_t_Qij(5,0) = Qij_t_Qij(0,5);
        
    Qij_t_Qij(2,1) = Qij_t_Qij(1,2);
    Qij_t_Qij(3,1) = Qij_t_Qij(1,3);
    Qij_t_Qij(4,1) = Qij_t_Qij(1,4);
    Qij_t_Qij(5,1) = Qij_t_Qij(1,5);
        
    Qij_t_Qij(3,2) = Qij_t_Qij(2,3);
    Qij_t_Qij(4,2) = Qij_t_Qij(2,4);
    Qij_t_Qij(5,2) = Qij_t_Qij(2,5);

    Qij_t_Qij(4,3) = Qij_t_Qij(3,4);
    Qij_t_Qij(5,3) = Qij_t_Qij(3,5);
        
    Qij_t_Qij(5,4) = Qij_t_Qij(4,5);
};
inline void SparseBundleAdjustmentSolver::calc_Qij_t_Qij_weight(const _BA_Numeric weight, const _BA_Mat26& Qij,
    _BA_Mat66& Qij_t_Qij)
{
    Qij_t_Qij.setZero();

    // a(0,1) = 0;
    // a(1,0) = 0;

    // Calculate upper triangle
    const _BA_Mat26& a = Qij;
    const _BA_Mat26 wa = weight*Qij;

    Qij_t_Qij(0,0) = (wa(0,0)*a(0,0)                 );
    // Qij_t_Qij(0,1) = (0);
    Qij_t_Qij(0,2) = (wa(0,0)*a(0,2)                 );
    Qij_t_Qij(0,3) = (wa(0,0)*a(0,3)                 );
    Qij_t_Qij(0,4) = (wa(0,0)*a(0,4)                 );
    Qij_t_Qij(0,5) = (wa(0,0)*a(0,5)                 );
    
    Qij_t_Qij(1,1) = (                 wa(1,1)*a(1,1));
    Qij_t_Qij(1,2) = (                 wa(1,1)*a(1,2));
    Qij_t_Qij(1,3) = (                 wa(1,1)*a(1,3));
    Qij_t_Qij(1,4) = (                 wa(1,1)*a(1,4));
    Qij_t_Qij(1,5) = (                 wa(1,1)*a(1,5));
    
    Qij_t_Qij(2,2) = (wa(0,2)*a(0,2) + wa(1,2)*a(1,2));
    Qij_t_Qij(2,3) = (wa(0,2)*a(0,3) + wa(1,2)*a(1,3));
    Qij_t_Qij(2,4) = (wa(0,2)*a(0,4) + wa(1,2)*a(1,4));
    Qij_t_Qij(2,5) = (wa(0,2)*a(0,5) + wa(1,2)*a(1,5));

    Qij_t_Qij(3,3) = (wa(0,3)*a(0,3) + wa(1,3)*a(1,3));
    Qij_t_Qij(3,4) = (wa(0,3)*a(0,4) + wa(1,3)*a(1,4));
    Qij_t_Qij(3,5) = (wa(0,3)*a(0,5) + wa(1,3)*a(1,5));
    
    Qij_t_Qij(4,4) = (wa(0,4)*a(0,4) + wa(1,4)*a(1,4));
    Qij_t_Qij(4,5) = (wa(0,4)*a(0,5) + wa(1,4)*a(1,5));
    
    Qij_t_Qij(5,5) = (wa(0,5)*a(0,5) + wa(1,5)*a(1,5));

    // Qij_t_Qij(0,0) = weight*(a(0,0)*a(0,0) + a(1,0)*a(1,0));
    // Qij_t_Qij(0,1) = weight*(a(0,0)*a(0,1) + a(1,0)*a(1,1));
    // Qij_t_Qij(0,2) = weight*(a(0,0)*a(0,2) + a(1,0)*a(1,2));
    // Qij_t_Qij(0,3) = weight*(a(0,0)*a(0,3) + a(1,0)*a(1,3));
    // Qij_t_Qij(0,4) = weight*(a(0,0)*a(0,4) + a(1,0)*a(1,4));
    // Qij_t_Qij(0,5) = weight*(a(0,0)*a(0,5) + a(1,0)*a(1,5));
    
    // Qij_t_Qij(1,1) = weight*(a(0,1)*a(0,1) + a(1,1)*a(1,1));
    // Qij_t_Qij(1,2) = weight*(a(0,1)*a(0,2) + a(1,1)*a(1,2));
    // Qij_t_Qij(1,3) = weight*(a(0,1)*a(0,3) + a(1,1)*a(1,3));
    // Qij_t_Qij(1,4) = weight*(a(0,1)*a(0,4) + a(1,1)*a(1,4));
    // Qij_t_Qij(1,5) = weight*(a(0,1)*a(0,5) + a(1,1)*a(1,5));
    
    // Qij_t_Qij(2,2) = weight*(a(0,2)*a(0,2) + a(1,2)*a(1,2));
    // Qij_t_Qij(2,3) = weight*(a(0,2)*a(0,3) + a(1,2)*a(1,3));
    // Qij_t_Qij(2,4) = weight*(a(0,2)*a(0,4) + a(1,2)*a(1,4));
    // Qij_t_Qij(2,5) = weight*(a(0,2)*a(0,5) + a(1,2)*a(1,5));

    // Qij_t_Qij(3,3) = weight*(a(0,3)*a(0,3) + a(1,3)*a(1,3));
    // Qij_t_Qij(3,4) = weight*(a(0,3)*a(0,4) + a(1,3)*a(1,4));
    // Qij_t_Qij(3,5) = weight*(a(0,3)*a(0,5) + a(1,3)*a(1,5));
    
    // Qij_t_Qij(4,4) = weight*(a(0,4)*a(0,4) + a(1,4)*a(1,4));
    // Qij_t_Qij(4,5) = weight*(a(0,4)*a(0,5) + a(1,4)*a(1,5));
    
    // Qij_t_Qij(5,5) = weight*(a(0,5)*a(0,5) + a(1,5)*a(1,5));
    

    // Substitute symmetric elements
    // Qij_t_Qij(1,0) = Qij_t_Qij(0,1);
    Qij_t_Qij(2,0) = Qij_t_Qij(0,2);
    Qij_t_Qij(3,0) = Qij_t_Qij(0,3);
    Qij_t_Qij(4,0) = Qij_t_Qij(0,4);
    Qij_t_Qij(5,0) = Qij_t_Qij(0,5);
        
    Qij_t_Qij(2,1) = Qij_t_Qij(1,2);
    Qij_t_Qij(3,1) = Qij_t_Qij(1,3);
    Qij_t_Qij(4,1) = Qij_t_Qij(1,4);
    Qij_t_Qij(5,1) = Qij_t_Qij(1,5);
        
    Qij_t_Qij(3,2) = Qij_t_Qij(2,3);
    Qij_t_Qij(4,2) = Qij_t_Qij(2,4);
    Qij_t_Qij(5,2) = Qij_t_Qij(2,5);

    Qij_t_Qij(4,3) = Qij_t_Qij(3,4);
    Qij_t_Qij(5,3) = Qij_t_Qij(3,5);
        
    Qij_t_Qij(5,4) = Qij_t_Qij(4,5);
};