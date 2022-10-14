#include "core/ba_solver/sparse_bundle_adjustment_scale_sqp.h"

SparseBundleAdjustmentScaleSQPSolver::SparseBundleAdjustmentScaleSQPSolver()
{

};

void SparseBundleAdjustmentScaleSQPSolver::setBAParametersAndConstraints(const std::shared_ptr<SparseBAParameters>& ba_params, const std::shared_ptr<ScaleConstraints>& scale_const)
{
    // BA parameters
    ba_params_ = ba_params;
    
    N_     = ba_params_->getNumOfAllFrames(); // Fixed frame == 0-th frame.
    N_opt_ = ba_params_->getNumOfOptimizeFrames(); // Opt. frames == all frames except for 0-th frame. Thus, N_opt == N - 1
    M_     = ba_params_->getNumOfOptimizeLandmarks(); // M
    n_obs_ = ba_params_->getNumOfObservations(); // OK


    // Constraints
    scale_const_ = scale_const;

    K_     = scale_const_->getNumOfConstraint();

    // Make storage fit!
    this->makeStorageSizeToFit();
};

void SparseBundleAdjustmentScaleSQPSolver::makeStorageSizeToFit()
{
    // Resize storages.
    A_.resize(N_opt_); 
    
    B_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j) 
        B_[j].resize(M_, _BA_Mat33::Zero());   // 6x3, N_opt X M blocks
    
    Bt_.resize(M_);    
    for(int i = 0; i < M_; ++i) 
        Bt_[i].resize(N_opt_, _BA_Mat33::Zero());   // 3x6, N_opt X M blocks
    
    C_.resize(M_);

    D_.resize(K_);
    for(int k = 0; k < K_; ++k)
        D_[k].resize(N_opt_, _BA_Mat13::Zero());

    Dt_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j)
        Dt_[j].resize(K_, _BA_Mat31::Zero());

    a_.resize(N_opt_); // 3x1, N_opt blocks
    b_.resize(M_);     // 3x1, M blocks
    c_.resize(K_);     // 1x1, K

    x_.resize(N_opt_); // 3x1, N_opt blocks
    y_.resize(M_);     // 3x1, M blocks
    z_.resize(K_);     // 1x1, K
    
    fixparams_rot_.resize(N_opt_);
    params_trans_.resize(N_opt_); // 3x1, N_opt blocks
    params_points_.resize(M_);    // 3x1, M blocks
    params_lagrange_.resize(K_);  // 1x1, K_




    Cinv_.resize(M_); // 3x3, M diagonal blocks 

    Cinvb_.resize(M_);

    CinvBt_.resize(M_);
    for(int i = 0; i < M_; ++i)
        CinvBt_[i].resize(N_opt_, _BA_Mat33::Zero());

    CinvBtx_.resize(M_);

    BCinv_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j)
        BCinv_[j].resize(M_, _BA_Mat33::Zero());

    BCinvb_.resize(N_opt_); // 3x1, N_opt x 1 blocks

    BCinvBt_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        BCinvBt_[j].resize(N_opt_, _BA_Mat33::Zero());




    Am_BCinvBt_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        Am_BCinvBt_[j].resize(N_opt_, _BA_Mat33::Zero());
    
    inv_Am_BCinvBt_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        inv_Am_BCinvBt_[j].resize(N_opt_, _BA_Mat33::Zero());
    
    am_BCinvb_.resize(N_opt_); // 3x1, N_opt x 1 blocks

    am_BCinvbm_Dtz_.resize(N_opt_);
};



void SparseBundleAdjustmentScaleSQPSolver::zeroizeStorageMatrices()
{
    // std::cout << "in zeroize \n";
    for(_BA_Index j = 0; j < N_opt_; ++j)
    {
        A_[j].setZero();
        a_[j].setZero();
        x_[j].setZero();
        BCinvb_[j].setZero();
        am_BCinvb_[j].setZero();
        am_BCinvbm_Dtz_[j].setZero();

        for(_BA_Index i = 0; i < M_; ++i){
            B_[j][i].setZero();
            Bt_[i][j].setZero();
            BCinv_[j][i].setZero();
            CinvBt_[i][j].setZero();
        }
        for(_BA_Index jj = 0; jj < N_opt_; ++jj)
        {
            BCinvBt_[j][jj].setZero();
            Am_BCinvBt_[j][jj].setZero();
            inv_Am_BCinvBt_[j][jj].setZero();
        }
        for(_BA_Index k = 0; k < K_; ++k)
        {
            D_[k][j].setZero();
            Dt_[j][k].setZero();
        }
    }

    for(_BA_Index i = 0; i < M_; ++i)
    {
        C_[i].setZero();
        b_[i].setZero();
        y_[i].setZero();
        Cinv_[i].setZero();
        Cinvb_[i].setZero();
        CinvBtx_[i].setZero();
    }

    for(_BA_Index k = 0; k < K_; ++k)
    {
        c_[k] = 0;
        z_[k] = 0;

    }
    // std::cout << "zeroize done\n";
};    

void SparseBundleAdjustmentScaleSQPSolver::setHuberThreshold(float thres_huber) { THRES_HUBER_ = thres_huber; };
void SparseBundleAdjustmentScaleSQPSolver::setCamera(const std::shared_ptr<Camera>& cam) { cam_ = cam; };
    
bool SparseBundleAdjustmentScaleSQPSolver::solveForFiniteIterations(int MAX_ITER)
{

    bool flag_success = true;

    // Set the Parameter Vector.
    setParameterVectorFromPosesPoints();

    // Do Iterations
    // Initialize parameters
    std::vector<_BA_numeric> r_prev(n_obs_, 0.0f);
    _BA_numeric err_prev = 1e10f;
    _BA_numeric lambda   = 0.0000001;
    for(int iter = 0; iter < MAX_ITER; ++iter)
    {
        // Reset A, B, Bt, C, Cinv, a, b, x, y...
        zeroizeStorageMatrices();

        // Iteratively solve. (Levenberg-Marquardt algorithm)
        int cnt = 0;
        _BA_numeric err = 0.0f;
        
        // For i-th landmark
        for(_BA_Index i = 0; i < M_; ++i)
        {           
            const LandmarkBA&   lmba = ba_params_->getLandmarkBA(i);
            const _BA_Point&    Xi   = lmba.X; 
            const FramePtrVec&  kfs  = lmba.kfs_seen;
            const _BA_PixelVec& pts  = lmba.pts_on_kfs;


            // For j-th landmark
            for(int jj = 0; jj < kfs.size(); ++jj)
            {               
                const _BA_Pixel& pij = pts.at(jj);
                const FramePtr&   kf = kfs.at(jj);

                // 0) check whether it is optimizable frame
                // three states: 1) non-opt (the first frame), 2) opt, 3) opt & constraint
                bool is_optimizable_frame = false;
                _BA_Index j  = -1; // optimization index
                if(ba_params_->isOptFrame(kf)){
                    is_optimizable_frame = true;
                    j = ba_params_->getOptPoseIndex(kf);
                }

                bool is_constrained_frame = false;
                _BA_Index jm = -1; // j_minor
                _BA_Index k  = -1; // constraint index
                if(scale_const_->isConstrainedFrame(kf)){
                    is_constrained_frame = true;
                    k = scale_const_->getConstIndexByMajorFrame(kf); // 해당 키프레임이 몇번 constraint와 관련이 있는가?
                    // 관련이 있다면, major frame인가 prev. frame인가를 판단해야함.
                    // Twj - Tw(j-1) 이 모두 관련있기 때문이다.
                    jm = scale_const_->isConstrainedFrame(kf);
                    // jm = 
                }
                
                // std::cout << "i,j,k: " << i << ", " << j << ", " << k << std::endl;

                // Get current camera parameters
                const _BA_numeric& fx = cam_->fx(), fy = cam_->fy();
                const _BA_numeric& cx = cam_->cx(), cy = cam_->cy();

                // Get poses
                const _BA_PoseSE3& T_jw = ba_params_->getPose(kf);
                const _BA_Rot3& R_jw = T_jw.block<3,3>(0,0);
                const _BA_Pos3& t_jw = T_jw.block<3,1>(0,3);

                _BA_Point Xij = R_jw*Xi + t_jw;

                // 1) Qij and Rij calculation
                const _BA_numeric& xj = Xij(0), yj = Xij(1), zj = Xij(2);
                _BA_numeric invz = 1.0/zj; _BA_numeric invz2 = invz*invz;

                _BA_numeric fxinvz      = fx*invz;      _BA_numeric fyinvz      = fy*invz;
                _BA_numeric xinvz       = xj*invz;      _BA_numeric yinvz       = yj*invz;
                _BA_numeric fx_xinvz2   = fxinvz*xinvz; _BA_numeric fy_yinvz2   = fyinvz*yinvz;
                _BA_numeric xinvz_yinvz = xinvz*yinvz;

                _BA_Mat23 Rij;
                const _BA_numeric& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2);
                const _BA_numeric& r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2);
                const _BA_numeric& r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
                Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                       fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33; // Related to dr/dXi

                // 2) residual calculation
                _BA_Pixel ptw;
                ptw(0) = fx*xinvz + cx;
                ptw(1) = fy*yinvz + cy;
                _BA_Vec2 rij;
                rij = ptw - pij;

                // 3) HUBER weight calculation (Manhattan distance)
                _BA_numeric absrxry = abs(rij(0))+abs(rij(1));
                r_prev[cnt] = absrxry;
                // std::cout << cnt << "-th point: " << absrxry << " [px]\n";

                _BA_numeric weight = 1.0;
                bool flag_weight = false;
                if(absrxry > THRES_HUBER_){
                    weight = (THRES_HUBER_/absrxry);
                    flag_weight = true;
                }

                // 4) Add (or fill) data (JtWJ & mJtWr & err).
            }
        }
    }


    std::cout << "End of optimization.\n";


    return flag_success;
};


void SparseBundleAdjustmentScaleSQPSolver::reset()
{
    ba_params_ = nullptr;
    scale_const_ = nullptr;
    
    cam_   = nullptr;

    N_     = 0;
    N_opt_ = 0;
    M_     = 0;
    K_     = 0;
    n_obs_ = 0;

    THRES_HUBER_ = 0;
    THRES_EPS_   = 0;


    A_.resize(0);
    B_.resize(0);
    Bt_.resize(0);
    C_.resize(0);
    D_.resize(0);
    Dt_.resize(0);
    
    a_.resize(0);
    b_.resize(0);
    c_.resize(0);

    x_.resize(0);
    y_.resize(0);
    z_.resize(0);

    fixparams_rot_.resize(0);
    params_trans_.resize(0);
    params_points_.resize(0);
    params_lagrange_.resize(0);

    Cinv_.resize(0);
    Cinvb_.resize(0);
    CinvBt_.resize(0);
    CinvBtx_.resize(0);
    BCinv_.resize(0);
    BCinvb_.resize(0);
    BCinvBt_.resize(0);

    Am_BCinvBt_.resize(0);
    inv_Am_BCinvBt_.resize(0);
    am_BCinvb_.resize(0);
    am_BCinvbm_Dtz_.resize(0);

    D_inv_Am_BCinvBt_Dt_.resize(0);

    std::cout << "Reset SQP solver.\n";
};


void SparseBundleAdjustmentScaleSQPSolver::setParameterVectorFromPosesPoints(){
    // 1) Pose part
    for(_BA_Index j_opt = 0; j_opt < N_opt_; ++j_opt)
    {
        const _BA_PoseSE3& T_jw = ba_params_->getOptPose(j_opt);
        const _BA_Rot3&    R_jw = T_jw.block<3,3>(0,0);
        const _BA_Pos3&    t_jw = T_jw.block<3,1>(0,3);
        fixparams_rot_[j_opt] = R_jw;
        params_trans_[j_opt]  = t_jw;
        // std::cout << "Pose:\n" << Tjw << std::endl;
        // std::cout << "xi_jw: " << xi_jw.transpose() << std::endl;
    }

    // 2) Point part
    for(_BA_Index i = 0; i < M_; ++i)
        params_points_[i] = ba_params_->getOptPoint(i);
};

void SparseBundleAdjustmentScaleSQPSolver::getPosesPointsFromParameterVector(){
    // Generate parameters
    // xi part 0~5, 6~11, ... 
    for(_BA_Index j_opt = 0; j_opt < N_opt_; ++j_opt){
        if(std::isnan(params_trans_[j_opt].norm()))
        {
            std::cerr << "std::isnan params_poses !\n";
            std::cout << params_trans_[j_opt] << std::endl;
        }
            
        _BA_PoseSE3 Tjw ;
        Tjw << fixparams_rot_[j_opt], params_trans_[j_opt],0,0,0,1;
        
        ba_params_->updateOptPose(j_opt, Tjw);
    }
    // point part
    for(_BA_Index i = 0; i < M_; ++i)
        ba_params_->updateOptPoint(i, params_points_[i]);
};