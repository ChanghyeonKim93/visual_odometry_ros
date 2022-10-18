#include "core/ba_solver/sparse_bundle_adjustment_scale_sqp.h"

SparseBundleAdjustmentScaleSQPSolver::SparseBundleAdjustmentScaleSQPSolver()
{
    std::cerr << "SparseBundleAdjustmentScaleSQPSolver is constructed.\n";
};

void SparseBundleAdjustmentScaleSQPSolver::setHuberThreshold(float thres_huber) 
{ 
    THRES_HUBER_ = thres_huber;
    std::cout << "SBASQP : THRES_HUBER: " << THRES_HUBER_ << std::endl;
};

void SparseBundleAdjustmentScaleSQPSolver::setCamera(const std::shared_ptr<Camera>& cam) 
{ 
    cam_ = cam; 
    std::cout << "SBASQP : camera is set." << std::endl;
};
    
void SparseBundleAdjustmentScaleSQPSolver::setBAParametersAndConstraints(const std::shared_ptr<SparseBAParameters>& ba_params, const std::shared_ptr<ScaleConstraints>& scale_const)
{
    // Get BA parameters
    ba_params_   = ba_params;

    // Get Constraints
    scale_const_ = scale_const;
    
    // Get BA parameter sizes & constraint sizes
    N_     = ba_params_->getNumOfAllFrames(); // Fixed frame == 0-th frame.
    N_opt_ = ba_params_->getNumOfOptimizeFrames(); // Opt. frames == all frames except for 0-th frame. Thus, N_opt == N - 1
    M_     = ba_params_->getNumOfOptimizeLandmarks(); // M
    n_obs_ = ba_params_->getNumOfObservations(); // OK

    K_     = scale_const_->getNumOfConstraint(); // # of constraints

    // Make storage fit to BA param sizes and constraints sizes.
    this->makeStorageSizeToFit();

    std::cout << "SBASQP : setBAParams and Constraints OK.\n";
    std::cout << "     N : " << N_ << std::endl;
    std::cout << " N_opt : " << N_opt_ << std::endl;
    std::cout << "     M : " << M_ << std::endl;
    std::cout << " n_obs : " << n_obs_ << std::endl;
    std::cout << "     K : " << K_ << "\n" << std::endl;
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

    BCinv_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j)
        BCinv_[j].resize(M_, _BA_Mat33::Zero());

    BCinvb_.resize(N_opt_); // 3x1, N_opt x 1 blocks

    BCinvBt_.resize(N_opt_); 
    for(int j = 0; j < N_opt_; ++j) 
        BCinvBt_[j].resize(N_opt_, _BA_Mat33::Zero());

    Ap_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j)
        Ap_[j].resize(N_opt_, _BA_Mat33::Zero());

    ap_.resize(N_opt_); // 3x1, N_opt x 1 blocks

    Apinv_ap_.resize(N_opt_); // 3x1, N_opt x 1 blocks

    Apinv_Dt_.resize(N_opt_);
    for(int j = 0; j < N_opt_; ++j)
        Apinv_Dt_[j].resize(K_, _BA_Mat31::Zero());

    D_Apinv_ap_.resize(K_);

    D_Apinv_Dt_.resize(K_);
    for(int k = 0; k < K_; ++k)
        D_Apinv_Dt_[k].resize(K_, 0);
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
        ap_[j].setZero();
        Apinv_ap_[j].setZero();

        for(_BA_Index i = 0; i < M_; ++i){
            B_[j][i].setZero();
            Bt_[i][j].setZero();
            BCinv_[j][i].setZero();
        }
        for(_BA_Index jj = 0; jj < N_opt_; ++jj)
        {
            BCinvBt_[j][jj].setZero();
            Ap_[j][jj].setZero();
        }
        for(_BA_Index k = 0; k < K_; ++k)
        {
            D_[k][j].setZero();
            Dt_[j][k].setZero();
            Apinv_Dt_[j][k].setZero();
        }
    }

    for(_BA_Index i = 0; i < M_; ++i)
    {
        C_[i].setZero();
        b_[i].setZero();
        y_[i].setZero();
        Cinv_[i].setZero();
        Cinvb_[i].setZero();
    }

    for(_BA_Index k = 0; k < K_; ++k)
    {
        c_[k] = 0;
        z_[k] = 0;
        D_Apinv_ap_[k] = 0;

        for(_BA_Index kk = 0; kk < K_; ++kk)
            D_Apinv_Dt_[k][kk] = 0;
    }
    // std::cout << "zeroize done\n";
};    

bool SparseBundleAdjustmentScaleSQPSolver::solveForFiniteIterations(int MAX_ITER)
{

    bool flag_success = true;

    std::cout << "SBASQP : start to solve iteratively...\n";

    // Set the Parameter Vector.
    this->setParameterVectorFromPosesPoints();


    // Do Iterations
    // Initialize parameters
    std::vector<_BA_numeric> r_prev(n_obs_, 0.0f);
    _BA_numeric err_prev = 1e10f;
    _BA_numeric lambda   = 0.0000001;
    std::cout << "start iteration\n";
    for(int iter = 0; iter < MAX_ITER; ++iter)
    {
        // Reset A, B, Bt, C, Cinv, a, b, x, y...
        this->zeroizeStorageMatrices();

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
                _BA_Index j = -1; // optimization index
                bool is_optimizable_frame = ba_params_->isOptFrame(kf);
                if(is_optimizable_frame) 
                    j = ba_params_->getOptPoseIndex(kf);
                
                // Get current camera parameters
                const _BA_numeric& fx = cam_->fx(), fy = cam_->fy(), cx = cam_->cx(), cy = cam_->cy();

                // Get poses
                const _BA_PoseSE3& T_jw = ba_params_->getPose(kf);
                const _BA_Rot3& R_jw = T_jw.block<3,3>(0,0);
                const _BA_Pos3& t_jw = T_jw.block<3,1>(0,3);

                _BA_Point Xij = R_jw*Xi + t_jw;

                const _BA_numeric& xj = Xij(0), yj = Xij(1), zj = Xij(2);
                _BA_numeric invz = 1.0/zj; _BA_numeric invz2 = invz*invz;

                _BA_numeric fxinvz      = fx*invz;      _BA_numeric fyinvz      = fy*invz;
                _BA_numeric xinvz       = xj*invz;      _BA_numeric yinvz       = yj*invz;
                _BA_numeric fx_xinvz2   = fxinvz*xinvz; _BA_numeric fy_yinvz2   = fyinvz*yinvz;
                _BA_numeric xinvz_yinvz = xinvz*yinvz;

                // 1) residual calculation
                _BA_Pixel ptw;
                ptw << fx*xinvz + cx, fy*yinvz + cy;
                _BA_Vec2 rij;
                rij = ptw - pij;

                // 2) HUBER weight calculation (Manhattan distance)
                _BA_numeric absrxry = abs(rij(0)) + abs(rij(1));
                r_prev[cnt] = absrxry;

                _BA_numeric weight = 1.0;
                bool flag_weight = false;
                if(absrxry > THRES_HUBER_){
                    weight = (THRES_HUBER_/absrxry);
                    flag_weight = true;
                }                
                // std::cout << cnt << "-th point: " << absrxry << " [px]\n";

                /*
                    Rij = d{\pi{p_i,xi_jw}}/d{Xi}   = dp/dw * dw/dXi
                    Qij = d{\pi(p_i,xi_jw)}/d{tjw}  = dp/dw * dw/dtjw
                */

                // 3) Rij calculation (Jacobian w.r.t. Xi)
                // d{p_ij}/d{X_i} = dp/dw  \in R^{2x3}
                _BA_Mat23 Rij;
                const _BA_numeric& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2),
                                   r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2),
                                   r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
                Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                       fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33; // Related to dr/dXi

                _BA_Mat33 Rij_t_Rij; // fixed pose
                _BA_Vec3  Rij_t_rij; // fixed pose
                if(flag_weight){
                    // Rij_t_Rij = weight * (Rij.transpose()*Rij);
                    this->calc_Rij_t_Rij_weight(weight, Rij, Rij_t_Rij);
                    Rij_t_rij = weight * (Rij.transpose()*rij);
                }
                else
                {
                    // Rij_t_Rij = Rij.transpose()*Rij;
                    this->calc_Rij_t_Rij(Rij, Rij_t_Rij);
                    Rij_t_rij = Rij.transpose()*rij;
                }

                C_[i].noalias() +=  Rij_t_Rij; // FILL STORAGE (3)
                b_[i].noalias() += -Rij_t_rij; // FILL STORAGE (5)      
                // 4) Qij calculation (Jacobian w.r.t. pose (trans))
                // 5) Add (or fill) data (JtWJ & mJtWr & err).
                // d{p_ij}/d{t_jw} = dp/dw * R_jw  \in R^{2x3}
                if(is_optimizable_frame) { // Optimizable keyframe.
                    _BA_Mat23 Qij;
                    Qij << fxinvz,0,-fx_xinvz2,
                           0,fyinvz,-fy_yinvz2;
                           
                    _BA_Mat33 Qij_t_Qij; // fixed pose, opt. pose
                    _BA_Mat33 Qij_t_Rij; // fixed pose, opt. pose
                    _BA_Vec3  Qij_t_rij; // fixed pose, opt. pose
                    if(flag_weight){
                        // Qij_t_Qij = weight * (Qij.transpose()*Qij );
                        this->calc_Qij_t_Qij_weight(weight, Qij, Qij_t_Qij);
                        Qij_t_Rij = weight * (Qij.transpose()*Rij );
                        Qij_t_rij = weight * (Qij.transpose()*rij );
                    }
                    else
                    {
                        // Qij_t_Qij = Qij.transpose()*Qij; // fixed pose, opt. pose
                        this->calc_Qij_t_Qij(Qij, Qij_t_Qij);
                        Qij_t_Rij = Qij.transpose()*Rij; // fixed pose, opt. pose
                        Qij_t_rij = Qij.transpose()*rij; // fixed pose, opt. pose                        
                    }

                    A_[j].noalias() +=  Qij_t_Qij; // FILL STORAGE (1)
                    B_[j][i]         =  Qij_t_Rij; // FILL STORAGE (2)
                    Bt_[i][j]        =  Qij_t_Rij.transpose().eval(); // FILL STORAGE (2-1)
                    a_[j].noalias() += -Qij_t_rij; // FILL STORAGE (4)

                    if(std::isnan(Qij_t_Qij.norm()) || std::isnan(Qij_t_Rij.norm()) || std::isnan(Qij_t_rij.norm()) )
                    {
                        throw std::runtime_error("In BASQP, pose becomes nan!");
                    }
                } // END is_optimizable_frame

                _BA_numeric err_tmp = weight*rij.transpose()*rij;
                err += err_tmp;

                ++cnt;
            } // END jj of i-th point
        } // END i-th point
        // From now, A, B, Bt, C, b are fully filled, and a is partially filled.

        // std::cout << "From now, A, B, Bt, C, b are fully filled, and a is partially filled.\n";

        // Manipulate constraints (D, Dt, a, c should be filled.)
        for(int k = 0; k < K_; ++k)
        {
            const RelatedFramePair& fp = scale_const_->getRelatedFramePairByConstraintIndex(k);
            const FramePtr& f_major = fp.getMajor();
            const FramePtr& f_minor = fp.getMinor();

            _BA_Scale s_k = scale_const_->getConstraintValueByMajorFrame(f_major);

            _BA_Index j1  = -1; // optimization index (major)
            _BA_Index j0  = -1; // optimization index (minor)
            bool is_major_optimizable = ba_params_->isOptFrame(f_major);
            if(is_major_optimizable){
                j1 = ba_params_->getOptPoseIndex(f_major);
            }
            bool is_minor_optimizable = ba_params_->isOptFrame(f_minor);
            if(is_minor_optimizable){
                j0 = ba_params_->getOptPoseIndex(f_minor);
            }

            // std::cout << k << "-th const. has frames: "
            //     << j1 <<"(" << f_major->getID() << ")" << "(" << (is_major_optimizable ? "O" : "X") << "), "
            //     << j0 <<"(" << f_minor->getID() << ")" << "(" << (is_minor_optimizable ? "O" : "X") << ")"
            //     << std::endl;

            // 1) for major frame
            const _BA_PoseSE3& T_1w = ba_params_->getPose(f_major);
            const _BA_Rot3&    R_1w = T_1w.block<3,3>(0,0);
            _BA_Pos3 t_w1   = -R_1w.transpose()*T_1w.block<3,1>(0,3);
            
            // 2) for minor frame
            const _BA_PoseSE3& T_0w = ba_params_->getPose(f_minor);
            const _BA_Rot3&    R_0w = T_0w.block<3,3>(0,0);
            _BA_Pos3 t_w0   = -R_0w.transpose()*T_0w.block<3,1>(0,3);

            _BA_Pos3 dt_k = t_w1 - t_w0;

            _BA_Pos3 dt_k1 = -2*R_1w*dt_k;
            _BA_Pos3 dt_k0 =  2*R_0w*dt_k;

            if(is_major_optimizable) {
                D_[k][j1]  = dt_k1.transpose();
                Dt_[j1][k] = dt_k1; 
                a_[j1].noalias() += -Dt_[j1][k]*params_lagrange_[k];
            }

            if(is_minor_optimizable) {
                D_[k][j0]  = dt_k0.transpose();
                Dt_[j0][k] = dt_k0;
                a_[j0].noalias() += -Dt_[j0][k]*params_lagrange_[k];
            }
            
            // a_[j].noalias() -= Dtk_lamk; // FILL STORAGE (4)
            // c_[k] -= D_[k];

            // Calculate g_k 
            _BA_numeric g_k = dt_k.transpose()*dt_k - s_k*s_k;
            c_[k] = -g_k;

        } // END k-th constraint
        // std::cout << "END k-th constraint\n";
        
        // Solve sequentially.
        // 1) Damping 'A_' diagonal
        for(_BA_Index j = 0; j < N_opt_; ++j)
        {
            A_[j](0,0) += lambda*A_[j](0,0);
            A_[j](1,1) += lambda*A_[j](1,1);
            A_[j](2,2) += lambda*A_[j](2,2);
        }
        // std::cout << "A damping done\n";

        // 2) Damping 'C_' diagonal, and Calculate 'Cinv_' & 'Cinvb_'
        for(int i = 0; i < M_; ++i)
        {
            // Damping
            C_[i](0,0) += lambda*C_[i](0,0);
            C_[i](1,1) += lambda*C_[i](1,1);
            C_[i](2,2) += lambda*C_[i](2,2);

            // Cinv and Cinvb
            Cinv_[i]  = C_[i].ldlt().solve(_BA_Mat33::Identity()); // inverse by ldlt
            Cinvb_[i] = Cinv_[i]*b_[i];  // FILL STORAGE (10)
        } 
        // std::cout << "C damping Cinv Cinvb done\n";

        // 3) Calculate 'BCinv_', 'BCinvb_',' BCinvBt_'
        for(int i = 0; i < M_; ++i)
        {
            const FramePtrVec&  kfs  = ba_params_->getLandmarkBA(i).kfs_seen;
            for(int jj = 0; jj < kfs.size(); ++jj) 
            {
                const FramePtr& kf = kfs[jj];

                bool is_optimizable_keyframe_j = false;
                _BA_Index j = -1;
                if(ba_params_->isOptFrame(kf)){
                    is_optimizable_keyframe_j = true;
                    j = ba_params_->getOptPoseIndex(kf);

                    BCinv_[j][i]  = B_[j][i]*Cinv_[i];
                    BCinvb_[j].noalias() += BCinv_[j][i]*b_[i];

                    for(int kk = jj; kk < kfs.size(); ++kk){
                        // For k-th keyframe
                        const FramePtr& kf2 = kfs[kk];
                        bool is_optimizable_keyframe_k = false;
                        _BA_Index k = -1;
                        if(ba_params_->isOptFrame(kf2)){
                            is_optimizable_keyframe_k = true;
                            k = ba_params_->getOptPoseIndex(kf2);
                            BCinvBt_[j][k].noalias() += BCinv_[j][i]*Bt_[i][k];  // FILL STORAGE (7)
                        }
                    }
                }
            }
        }
        for(_BA_Index j = 0; j < N_opt_; ++j)
            for(_BA_Index u = j; u < N_opt_; ++u)
                BCinvBt_[u][j] = BCinvBt_[j][u].transpose().eval();
        // std::cout << "BCinvBt done\n";
            
        // 4) Calculate 'Ap_' and 'ap_'
        for(_BA_Index j = 0; j < N_opt_; ++j){
            ap_[j] = a_[j] - BCinvb_[j];
            for(_BA_Index u = 0; u < N_opt_; ++u){
                if(j == u) Ap_[j][j] = A_[j] - BCinvBt_[j][j];
                else       Ap_[j][u] =       - BCinvBt_[j][u];
            }
        }
        // std::cout << "ap Ap done\n";

        // 5) Solve 'Apinv_ap_' and 'Apinv_Dt_'
        // solve 'Apinv_ap_'
        _BA_MatX Ap_mat(3*N_opt_,3*N_opt_);
        _BA_MatX ap_mat(3*N_opt_,1);
        int idx0 = 0;
        for(_BA_Index j = 0; j < N_opt_; ++j, idx0 += 3){
            ap_mat.block(idx0,0,3,1) = ap_[j];
            int idx1 = 0;
            for(_BA_Index u = 0; u < N_opt_; ++u, idx1 += 3){
                Ap_mat.block(idx0,idx1,3,3) = Ap_[j][u];
            }
        }
        _BA_MatX Apinv_ap_mat = Ap_mat.ldlt().solve(ap_mat);
        idx0 = 0;
        for(_BA_Index j = 0; j < N_opt_; ++j, idx0 += 3)
            Apinv_ap_[j] = Apinv_ap_mat.block<3,1>(idx0,0);
        // std::cout << "Apinv_ap_ done\n";

        // solve 'Apinv_Dt_'
        _BA_MatX Dt_mat(3*N_opt_,K_);
        idx0 = 0;
        for(_BA_Index j = 0; j < N_opt_; ++j, idx0 += 3){
            for(_BA_Index k = 0; k < K_; ++k){
                const int& idx1 = k;
                Dt_mat.block(idx0,idx1,3,1) = Dt_[j][k];
            }
        }
        _BA_MatX Apinv_Dt_mat = Ap_mat.ldlt().solve(Dt_mat);
        idx0 = 0;
        for(_BA_Index j = 0; j < N_opt_; ++j, idx0 += 3){
            for(_BA_Index k = 0; k < K_; ++k)
               Apinv_Dt_[j][k] = Apinv_Dt_mat.block<3,1>(idx0,k);
        }
        // std::cout << "Apinv_Dt_ done\n";

        // 6) Solve 'D_Apinv_ap_' and 'D_Apinv_Dt_'
        for(int k = 0; k < K_; ++k )
            for(int j = 0; j < N_opt_; ++j)
                D_Apinv_ap_[k] += D_[k][j]*Apinv_ap_[j];

        for(int k = 0; k < K_; ++k )
            for(int kk = 0; kk < K_; ++kk)
                for(int j = 0; j < N_opt_; ++j)
                    D_Apinv_Dt_[k][kk] += D_[k][j]*Apinv_Dt_[j][kk];
        // std::cout << "'D_Apinv_ap_' and 'D_Apinv_Dt_' done\n";

        // 7) Solve the final problem by Schur Complement (solving order: z, x, y)
        // 7-1) Calculate z
        _BA_MatX D_Apinv_Dt_mat(K_,K_);
        _BA_MatX D_Apinv_ap_m_c_mat(K_,1);
        for(_BA_Index k = 0; k < K_; ++k) {
            D_Apinv_ap_m_c_mat(k) = D_Apinv_ap_[k] - c_[k];
            for(_BA_Index kk = 0; kk < K_; ++kk)
                D_Apinv_Dt_mat(k,kk) = D_Apinv_Dt_[k][kk];
        }
        _BA_MatX z_mat = D_Apinv_Dt_mat.ldlt().solve(D_Apinv_ap_m_c_mat);
        for(int k = 0; k < K_; ++k)
            z_[k] = z_mat(k);
        
        // 7-2) Calculate x
        _BA_MatX x_mat = Ap_mat.ldlt().solve(ap_mat-Dt_mat*z_mat);
        idx0 = 0;
        for(int j = 0; j < N_opt_; ++j, idx0 += 3)
            x_[j] = x_mat.block(idx0,0,3,1);

        // 7-3) Calculate y
        for(int i = 0; i < M_; ++i)
        {
            _BA_Vec3 Btx_tmp; Btx_tmp.setZero();
            for(int j = 0; j < N_opt_; ++j)
                Btx_tmp += (Bt_[i][j]*x_[j]);

            y_[i] = Cinv_[i] * (b_[i] - Btx_tmp);
        }

        // std::cout << "Pose update:\n";
        // for(int j = 0; j < N_opt_; ++j)
        //     std::cout << x_[j].transpose() << std::endl;
        // std::cout << "Lagrangian update:\n";
        // for(int k = 0; k < K_; ++k)
        //     std::cout << z_[k] << std::endl;
        // std::cout << "Constraint value:\n";
        // for(int k = 0; k < K_; ++k)
        //     std::cout << c_[k] << std::endl;

        // Update step
        for(_BA_Index j = 0; j < N_opt_; ++j)
            params_trans_[j].noalias() += x_[j];

        for(_BA_Index i = 0; i < M_; ++i)
            params_points_[i].noalias() += y_[i];

        for(_BA_Index k = 0; k < K_; ++k)
            params_lagrange_[k] += z_[k];

        this->getPosesPointsFromParameterVector();


        _BA_numeric average_error = 0.5*err/(_BA_numeric)n_obs_;
            
        std::cout << iter << "-th iter, error : " << average_error << "\n";

        // Check extraordinary cases.
        // flag_nan_pass   = std::isnan(err) ? false : true;
        // flag_error_pass = (average_error <= THRES_SUCCESS_AVG_ERROR) ? true : false;
        // flag_success    = flag_nan_pass && flag_error_pass;

    } // END iter-th iteration



    // Finally, update parameters
    if(1)
    {
        for(_BA_Index j = 0; j < N_opt_; ++j){
            const FramePtr& kf = ba_params_->getOptFramePtr(j);
            
            _BA_PoseSE3 Tjw_update = ba_params_->getPose(kf);
            Tjw_update = ba_params_->recoverOriginalScalePose(Tjw_update);
            Tjw_update = ba_params_->changeInvPoseRefToWorld(Tjw_update);

            PoseSE3 Tjw_update_float;
            Tjw_update_float << Tjw_update(0,0),Tjw_update(0,1),Tjw_update(0,2),Tjw_update(0,3),
                                Tjw_update(1,0),Tjw_update(1,1),Tjw_update(1,2),Tjw_update(1,3),
                                Tjw_update(2,0),Tjw_update(2,1),Tjw_update(2,2),Tjw_update(2,3),
                                Tjw_update(3,0),Tjw_update(3,1),Tjw_update(3,2),Tjw_update(3,3);
            kf->setPose(geometry::inverseSE3_f(Tjw_update_float));
        } 
        for(_BA_Index i = 0; i < M_; ++i){
            const LandmarkBA& lmba = ba_params_->getLandmarkBA(i);
            const LandmarkPtr& lm = lmba.lm;
            _BA_Point X_updated = lmba.X;

            X_updated = ba_params_->recoverOriginalScalePoint(X_updated);
            X_updated = ba_params_->warpToWorld(X_updated);
                       
            Point X_update_float;
            X_update_float << X_updated(0),X_updated(1),X_updated(2);

            lm->set3DPoint(X_update_float);
            lm->setBundled();
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


    Ap_.resize(0);
    ap_.resize(0); 

    Apinv_ap_.resize(0);
    Apinv_Dt_.resize(0);

    D_Apinv_ap_.resize(0);
    D_Apinv_Dt_.resize(0);

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
    }

    // 2) Point part
    for(_BA_Index i = 0; i < M_; ++i)
        params_points_[i] = ba_params_->getOptPoint(i);

    std::cout << "SBASQP : set poses and points OK\n";

    // 3) Lagragian part
    this->initializeLagrangeMultipliers();
    std::cout << "SBASQP : set constraints OK\n";
};

void SparseBundleAdjustmentScaleSQPSolver::initializeLagrangeMultipliers()
{
    // Find initial Lambda
    // Reset A, B, Bt, C, Cinv, a, b, x, y...
    this->zeroizeStorageMatrices(); // 저장 행렬 초기화..
    std::cout << "zeroize OK\n";
    
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
            _BA_Index j = -1; // optimization index
            bool is_optimizable_frame = ba_params_->isOptFrame(kf);
            if(is_optimizable_frame) 
                j = ba_params_->getOptPoseIndex(kf);
            
            // Get current camera parameters
            const _BA_numeric& fx = cam_->fx(), fy = cam_->fy(), cx = cam_->cx(), cy = cam_->cy();

            // Get poses
            const _BA_PoseSE3& T_jw = ba_params_->getPose(kf);
            const _BA_Rot3& R_jw = T_jw.block<3,3>(0,0);
            const _BA_Pos3& t_jw = T_jw.block<3,1>(0,3);

            _BA_Point Xij = R_jw*Xi + t_jw;

            const _BA_numeric& xj = Xij(0), yj = Xij(1), zj = Xij(2);
            _BA_numeric invz = 1.0/zj; _BA_numeric invz2 = invz*invz;

            _BA_numeric fxinvz      = fx*invz;      _BA_numeric fyinvz      = fy*invz;
            _BA_numeric xinvz       = xj*invz;      _BA_numeric yinvz       = yj*invz;
            _BA_numeric fx_xinvz2   = fxinvz*xinvz; _BA_numeric fy_yinvz2   = fyinvz*yinvz;
            _BA_numeric xinvz_yinvz = xinvz*yinvz;

            // 1) residual calculation
            _BA_Pixel ptw;
            ptw << fx*xinvz + cx, fy*yinvz + cy;
            _BA_Vec2 rij;
            rij = ptw - pij;

            // 2) HUBER weight calculation (Manhattan distance)
            _BA_numeric absrxry = abs(rij(0)) + abs(rij(1));

            _BA_numeric weight = 1.0;
            bool flag_weight = false;
            if(absrxry > THRES_HUBER_){
                weight = (THRES_HUBER_/absrxry);
                flag_weight = true;
            }                

            // 3) Rij calculation (Jacobian w.r.t. Xi)
            // d{p_ij}/d{X_i} = dp/dw  \in R^{2x3}
            _BA_Mat23 Rij;
            const _BA_numeric& r11 = R_jw(0,0), r12 = R_jw(0,1), r13 = R_jw(0,2),
                                r21 = R_jw(1,0), r22 = R_jw(1,1), r23 = R_jw(1,2),
                                r31 = R_jw(2,0), r32 = R_jw(2,1), r33 = R_jw(2,2);
            Rij << fxinvz*r11-fx_xinvz2*r31, fxinvz*r12-fx_xinvz2*r32, fxinvz*r13-fx_xinvz2*r33, 
                    fyinvz*r21-fy_yinvz2*r31, fyinvz*r22-fy_yinvz2*r32, fyinvz*r23-fy_yinvz2*r33; // Related to dr/dXi

            _BA_Vec3  Rij_t_rij = Rij.transpose()*rij; // fixed pose
            if(flag_weight){
                Rij_t_rij *= weight;
            }

            b_[i].noalias() += -Rij_t_rij; // FILL STORAGE (5)      
            // 4) Qij calculation (Jacobian w.r.t. pose (trans))
            // 5) Add (or fill) data (JtWJ & mJtWr & err).
            // d{p_ij}/d{t_jw} = dp/dw * R_jw  \in R^{2x3}
            if(is_optimizable_frame) { // Optimizable keyframe.
                _BA_Mat23 Qij;
                Qij << fxinvz,0,-fx_xinvz2,
                        0,fyinvz,-fy_yinvz2;
                        
                _BA_Vec3  Qij_t_rij = Qij.transpose()*rij; // fixed pose, opt. pose
                if(flag_weight){
                    Qij_t_rij *= weight;
                }
                a_[j].noalias() += -Qij_t_rij; // FILL STORAGE (4)

                if(std::isnan(Qij_t_rij.norm()) )
                {
                    throw std::runtime_error("In BASQP, pose becomes nan!");
                }
            } // END is_optimizable_frame
        } // END jj of i-th point
    } // END i-th point
    // From now, a, b are fully filled.

    // Manipulate constraints (D, Dt should be filled.)
    for(int k = 0; k < K_; ++k)
    {
        const RelatedFramePair& fp = scale_const_->getRelatedFramePairByConstraintIndex(k);
        const FramePtr& f_major = fp.getMajor();
        const FramePtr& f_minor = fp.getMinor();

        _BA_Scale s_k = scale_const_->getConstraintValueByMajorFrame(f_major);

        _BA_Index j1  = -1; // optimization index
        _BA_Index j0  = -1; // optimization index
        bool is_major_optimizable = ba_params_->isOptFrame(f_major);
        if(is_major_optimizable){
            j1 = ba_params_->getOptPoseIndex(f_major);
        }
        bool is_minor_optimizable = ba_params_->isOptFrame(f_minor);
        if(is_minor_optimizable){
            j0 = ba_params_->getOptPoseIndex(f_minor);
        }

        // 1) for major frame
        const _BA_PoseSE3& T_1w = ba_params_->getPose(f_major);
        const _BA_Rot3&    R_1w = T_1w.block<3,3>(0,0);
        _BA_Pos3 t_w1   = -R_1w.transpose()*T_1w.block<3,1>(0,3);
        
        // 2) for minor frame
        const _BA_PoseSE3& T_0w = ba_params_->getPose(f_minor);
        const _BA_Rot3&    R_0w = T_0w.block<3,3>(0,0);
        _BA_Pos3 t_w0   = -R_0w.transpose()*T_0w.block<3,1>(0,3);

        const _BA_Pos3& dt_k = t_w1 - t_w0;

        _BA_Mat31 dt_k1 = -2*R_1w*dt_k;
        _BA_Mat31 dt_k0 =  2*R_0w*dt_k;

        if(is_major_optimizable) {
            D_[k][j1]  = dt_k1.transpose();
            Dt_[j1][k] = dt_k1; 
        }

        if(is_minor_optimizable) {
            D_[k][j0]  = dt_k0.transpose();
            Dt_[j0][k] = dt_k0;
        }
    } // END k-th constraint


    // Solve D^t * lambda = [a;b]
    // D  : K x 3*N_opt
    // Dt : 3*N_opt x K
    // a : 3*N_opt x 1
    _BA_MatX D_mat(K_, 3*N_opt_);
    _BA_MatX a_mat(3*N_opt_, 1);

    for(int k = 0; k < K_; ++k)
    {
        int idx0 = k;
        for(int j = 0; j < N_opt_; ++j)
        {
            int idx1 = 3*j;
            D_mat.block(idx0,idx1,1,3) = D_[k][j];
        }
    }
    for(int j = 0; j < N_opt_; ++j)
    {
        int idx0 = 3*j;
        a_mat.block(idx0,0, 3,1) = a_[j];
    }
    
    _BA_MatX Dt_mat     = D_mat.transpose();
    _BA_MatX lambda_mat = (D_mat*Dt_mat).ldlt().solve(D_mat*a_mat);

    // std::cout << "lambda_mat.transpose():\n" 
    //     << lambda_mat.transpose() << std::endl;

    // Set params_lagrange_
    for(int k = 0; k < K_; ++k)
        params_lagrange_[k] = lambda_mat(k,0);
    
    std::cout << "params_lagrange:\n";
    for(int k = 0; k < K_; ++k)
        std::cout << params_lagrange_[k] << ",";
    
    std::cout << std::endl;
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








// For fast calculations for symmetric matrices
inline void SparseBundleAdjustmentScaleSQPSolver::calc_Rij_t_Rij(const _BA_Mat23& Rij, 
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
inline void SparseBundleAdjustmentScaleSQPSolver::calc_Rij_t_Rij_weight(const _BA_numeric weight, const _BA_Mat23& Rij,
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



inline void SparseBundleAdjustmentScaleSQPSolver::calc_Qij_t_Qij(const _BA_Mat23& Qij,
    _BA_Mat33& Qij_t_Qij)
{
    Qij_t_Qij.setZero();

    // a(0,1) = 0;
    // a(1,0) = 0;

    // Calculate upper triangle
    const _BA_Mat23& a = Qij;
    Qij_t_Qij(0,0) = (a(0,0)*a(0,0)                );
    Qij_t_Qij(0,1) = (a(0,0)*a(0,1)                );
    Qij_t_Qij(0,2) = (a(0,0)*a(0,2)                );
    
    Qij_t_Qij(1,1) = (                a(1,1)*a(1,1));
    Qij_t_Qij(1,2) = (                a(1,1)*a(1,2));
    
    Qij_t_Qij(2,2) = (a(0,2)*a(0,2) + a(1,2)*a(1,2));

    // Qij_t_Qij(0,0) = (a(0,0)*a(0,0) + a(1,0)*a(1,0));
    // Qij_t_Qij(0,1) = (a(0,0)*a(0,1) + a(1,0)*a(1,1));
    // Qij_t_Qij(0,2) = (a(0,0)*a(0,2) + a(1,0)*a(1,2));

    // Qij_t_Qij(1,1) = (a(0,1)*a(0,1) + a(1,1)*a(1,1));
    // Qij_t_Qij(1,2) = (a(0,1)*a(0,2) + a(1,1)*a(1,2));
    
    // Qij_t_Qij(2,2) = (a(0,2)*a(0,2) + a(1,2)*a(1,2));    

    // Substitute symmetric elements
    // Qij_t_Qij(1,0) = Qij_t_Qij(0,1);
    Qij_t_Qij(2,0) = Qij_t_Qij(0,2);
        
    Qij_t_Qij(2,1) = Qij_t_Qij(1,2);
};
inline void SparseBundleAdjustmentScaleSQPSolver::calc_Qij_t_Qij_weight(const _BA_numeric weight, const _BA_Mat23& Qij,
    _BA_Mat33& Qij_t_Qij)
{
    Qij_t_Qij.setZero();

    // a(0,1) = 0;
    // a(1,0) = 0;

    // Calculate upper triangle
    const _BA_Mat23& a = Qij;
    const _BA_Mat23 wa = weight*Qij;

    Qij_t_Qij(0,0) = (wa(0,0)*a(0,0)                 );
    // Qij_t_Qij(0,1) = (0);
    Qij_t_Qij(0,2) = (wa(0,0)*a(0,2)                 );
    
    Qij_t_Qij(1,1) = (                 wa(1,1)*a(1,1));
    Qij_t_Qij(1,2) = (                 wa(1,1)*a(1,2));
    
    Qij_t_Qij(2,2) = (wa(0,2)*a(0,2) + wa(1,2)*a(1,2));

    // Qij_t_Qij(0,0) = weight*(a(0,0)*a(0,0) + a(1,0)*a(1,0));
    // Qij_t_Qij(0,1) = weight*(a(0,0)*a(0,1) + a(1,0)*a(1,1));
    // Qij_t_Qij(0,2) = weight*(a(0,0)*a(0,2) + a(1,0)*a(1,2));
    
    // Qij_t_Qij(1,1) = weight*(a(0,1)*a(0,1) + a(1,1)*a(1,1));
    // Qij_t_Qij(1,2) = weight*(a(0,1)*a(0,2) + a(1,1)*a(1,2));
    
    // Qij_t_Qij(2,2) = weight*(a(0,2)*a(0,2) + a(1,2)*a(1,2));
    

    // Substitute symmetric elements
    // Qij_t_Qij(1,0) = Qij_t_Qij(0,1);
    Qij_t_Qij(2,0) = Qij_t_Qij(0,2);
        
    Qij_t_Qij(2,1) = Qij_t_Qij(1,2);
};