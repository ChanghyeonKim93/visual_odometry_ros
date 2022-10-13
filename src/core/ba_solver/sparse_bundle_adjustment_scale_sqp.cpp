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


bool SparseBundleAdjustmentScaleSQPSolver::solveForFiniteIterations(int MAX_ITER)
{


    // Do Iterations
    for(int iter = 0; iter < MAX_ITER; ++iter)
    {
        // Reset A, B, Bt, C, Cinv, a, b, x, y...
        zeroizeStorageMatrices();

        // Iteratively solve. (Levenberg-Marquardt algorithm)
        int cnt = 0;
        _BA_numeric err = 0.0f;
        for(_BA_Index i = 0; i < M_; ++i)
        {
            // For i-th landmark,
            const LandmarkBA&   lmba = ba_params_->getLandmarkBA(i);
            const _BA_Point&    Xi   = lmba.X; 
            const FramePtrVec&  kfs  = lmba.kfs_seen;
            const _BA_PixelVec& pts  = lmba.pts_on_kfs;

            
            for(int jj = 0; jj < kfs.size(); ++jj)
            {
                // For j-th landmark
                const _BA_Pixel& pij = pts.at(jj);
                const FramePtr&   kf = kfs.at(jj);

                // 0) check whether it is optimizable frame
                // three states: 1) non-opt (the first frame), 2) opt, 3) opt & constraint
                bool is_optimizable_frame = false;
                bool is_constrained_frame = false;
                _BA_Index j = -1; // optimization index
                _BA_Index k = -1; // constraint index
                if(ba_params_->isOptFrame(kf)){
                    is_optimizable_frame = true;
                    j = ba_params_->getOptPoseIndex(kf);
                }
                if(scale_const_->isConstrainedFrame(kf)){
                    is_constrained_frame = true;
                    // k = scale_const_->getConstIndex(kf); // 해당 키프레임이 몇번 constraint와 관련이 있는가?
                    // 관련이 있다면, major frame인가 prev. frame인가를 판단해야함.
                    // Twj - Tw(j-1) 이 모두 관련있기 때문이다.
                }
            }
        }
    }


};