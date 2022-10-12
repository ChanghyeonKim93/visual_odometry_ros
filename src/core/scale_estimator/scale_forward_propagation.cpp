#include "core/scale_estimator/scale_forward_propagation.h"

ScaleForwardPropagation::ScaleForwardPropagation(const std::shared_ptr<Camera>& cam)
: cam_(cam)
{

    // SFP parameters
    thres_age_past_horizon_ = 20;
    thres_age_use_          = 2;
    thres_age_recon_        = 15;
    thres_flow_             = 5.0;

    thres_parallax_use_     = 0.5*D2R;
    thres_parallax_recon_   = 60*D2R;

	// scale_estimator_->setTurnRegion_ThresCountTurn(params_.scale_estimator.thres_cnt_turns);
	// scale_estimator_->setTurnRegion_ThresPsi(params_.scale_estimator.thres_turn_psi*D2R);

	// scale_estimator_->setSFP_ThresAgePastHorizon(params_.scale_estimator.thres_age_past_horizon);
	// scale_estimator_->setSFP_ThresAgeUse(params_.scale_estimator.thres_age_use);
	// scale_estimator_->setSFP_ThresAgeRecon(params_.scale_estimator.thres_age_recon);
	// scale_estimator_->setSFP_ThresParallaxUse(params_.scale_estimator.thres_parallax_use*D2R);
	// scale_estimator_->setSFP_ThresParallaxRecon(params_.scale_estimator.thres_parallax_recon*D2R);


    std::cerr << "ScaleForwardPropagation is constructed.\n";
};

ScaleForwardPropagation::~ScaleForwardPropagation()
{
    std::cerr << "ScaleForwardPropagation is destructed.\n";
};

void ScaleForwardPropagation::solveLeastSquares_SFP(const SpMat& AtA, const SpVec& Atb, uint32_t M_tmp,
    SpVec& theta)
{
    uint32_t sz = 3*M_tmp;

    // AtA = [A, B; B.', C];
    // Atb = [a;b];
    // theta = [x;y];
    SpMat A = AtA.block(0,0, 3*M_tmp, 3*M_tmp);
    SpMat B = AtA.block(0,3*M_tmp, 3*M_tmp,1);

    float C = AtA.coeff(3*M_tmp, 3*M_tmp);

    // SpVec a = Atb.block<3*M_tmp,1>(0,0);
    SpVec a = Atb.block(0,0, 3*M_tmp,1);
    float b = Atb.coeff(3*M_tmp,1);

    // Calculate 'Ainv_vec'
    std::vector<Mat33> Ainv_vec; Ainv_vec.reserve(M_tmp);
    this->calcAinvVec_SFP(A, Ainv_vec, M_tmp);

    // Calculate AinvB ( == BtAinv.transpose())
    SpVec AinvB(sz,1);
    SpMat BtAinv(1,sz);

    this->calcAinvB_SFP(Ainv_vec, B, M_tmp, AinvB);
    BtAinv = AinvB.transpose().eval();
    
    SpVec x;
    float y;
    SpMat BtAinvB = (BtAinv*B);
    // std::cout << "BtAinvB size: " << BtAinvB.innerSize() << " x " << BtAinvB.outerSize() << std::endl;
    
    float CmBtAinvB = C-BtAinvB.coeff(0,0);
    float inv_CmBtAinvB = 1.0f/CmBtAinvB;
    SpMat BtAinv_a = BtAinv*a;
    // std::cout << "BtAinv_a size: " << BtAinv_a.innerSize() << " x " << BtAinv_a.outerSize() << std::endl;
    y = inv_CmBtAinvB*(b-BtAinv_a.coeff(0,0));
    SpVec amBy = a-B*y;
    SpVec Ainv_amBy;
    this->calcAinvB_SFP(Ainv_vec, amBy, M_tmp, Ainv_amBy);

    x = Ainv_amBy;
    std::cout << "SCALE : " << y << std::endl;
    theta.coeffRef(3*M_tmp) = y;
    for(int i = 0; i < M_tmp; ++i){
        theta.coeffRef(i) = x.coeff(i);
    }

};

void ScaleForwardPropagation::calcAinvVec_SFP(const SpMat& AA, std::vector<Mat33>& Ainv_vec, uint32_t M_tmp){
    Ainv_vec.reserve(M_tmp);
    Ainv_vec.resize(0);
    uint32_t idx[3] = {0,1,2};
    for(int i = 0; i < M_tmp; ++i){
        Mat33 Ainv_tmp = Mat33::Zero();
        Mat33 A_tmp = Mat33::Zero();

        A_tmp << AA.coeff(idx[0],idx[0]), AA.coeff(idx[0],idx[1]), AA.coeff(idx[0],idx[2]),
                    AA.coeff(idx[1],idx[0]), AA.coeff(idx[1],idx[1]), AA.coeff(idx[1],idx[2]),
                    AA.coeff(idx[2],idx[0]), AA.coeff(idx[2],idx[1]), AA.coeff(idx[2],idx[2]);
        Ainv_tmp = A_tmp.inverse();

        // float C = AA.coeff(idx[0],idx[0]);
        // float invC = 1.0f/C;
        // float A = AA.coeff(idx[0],idx[2]);
        // float B = AA.coeff(idx[1],idx[2]);
        // float D = AA.coeff(idx[2],idx[2]);

        // float CD = C*D;
        // float AB = A*B;
        // float AC = A*C;
        // float BC = B*C;
        // float AA = A*A;
        // float BB = B*B;
        // float CC = C*C;

        // float den = C*(AA + BB - CD);
        // // std::cout <<i <<" -th den: " << den << std::endl;
        // if(abs(den) < 1e-16 ) {
        //     std::cout << i << "-th point: " << C <<", " << AA <<", " << BB << ", " << CD << ":, den: " << abs(den) << std::endl;
        //     throw std::runtime_error("denominator is too small.");
        // }

        // den = 1.0f / den;

        // Ainv_tmp(0,0) = (BB - CD)*den;
        // Ainv_tmp(1,1) = (AA - CD)*den;
        // Ainv_tmp(2,2) = (-CC)*den;

        // Ainv_tmp(0,1) = (-AB)*den;
        // Ainv_tmp(0,2) = AC*den;
        // Ainv_tmp(1,2) = BC*den;

        // Ainv_tmp(1,0) = Ainv_tmp(0,1);
        // Ainv_tmp(2,0) = Ainv_tmp(0,2);
        // Ainv_tmp(2,1) = Ainv_tmp(1,2);
        
        Ainv_vec.push_back(Ainv_tmp);

        idx[0] += 3; idx[1] +=3; idx[2] += 3;
    }
};

void ScaleForwardPropagation::calcAinvB_SFP(const std::vector<Mat33>& Ainv_vec, const SpVec& B, uint32_t M_tmp,
     SpVec& AinvB)
{
    if(Ainv_vec.size() != M_tmp ) 
        throw std::runtime_error("Ainv_vec.size() != M_tmp.");

    AinvB.resize(3*M_tmp);

    int idx[3] = {0,1,2};
    for(int i = 0; i < M_tmp; ++i){
        const Eigen::Matrix3f& Ainv_part = Ainv_vec[i];

        Eigen::Vector3f B_part;
        B_part << B.coeff(idx[0],0), B.coeff(idx[1],0), B.coeff(idx[2],0);

        Eigen::Vector3f res_part;
        res_part = Ainv_part*B_part;
        
        AinvB.coeffRef(idx[0],0) = res_part(0);
        AinvB.coeffRef(idx[1],0) = res_part(1);
        AinvB.coeffRef(idx[2],0) = res_part(2);

        idx[0] += 3; idx[1] += 3; idx[2] += 3;
    }
};

void ScaleForwardPropagation::setSFP_ThresAgePastHorizon(uint32_t age_past_horizon){
    thres_age_past_horizon_ = age_past_horizon;
};
void ScaleForwardPropagation::setSFP_ThresAgeUse(uint32_t age_use){
    thres_age_use_ = age_use;
};
void ScaleForwardPropagation::setSFP_ThresAgeRecon(uint32_t age_recon){
    thres_age_recon_ = age_recon;
};
void ScaleForwardPropagation::setSFP_ThresParallaxUse(float thres_parallax_use){
    thres_parallax_use_ = thres_parallax_use;
};
void ScaleForwardPropagation::setSFP_ThresParallaxRecon(float thres_parallax_recon){
    thres_parallax_recon_ = thres_parallax_recon;
};


void ScaleForwardPropagation::runSFP(
    const LandmarkPtrVec& lmvec, 
    const FramePtrVec& framevec, 
    const PoseSE3& dT10)
{
    // intrinsic parameters
    const float& fx = cam_->fx(), fy = cam_->fy();
    const float& fxinv = cam_->fxinv(), fyinv = cam_->fyinv();
    const float& cx = cam_->cx(), cy = cam_->cy();
    const Eigen::Matrix3f& K    = cam_->K();
    const Eigen::Matrix3f& Kinv = cam_->Kinv();

    // Get current frame
    FramePtr frame_curr = framevec.back();
    uint32_t j = frame_curr->getID() ;

    // Pixels tracked in this frame

    // Get relative motion
    PoseSE3 dT = dT10;
    
    // Get rotation of this frame w.r.t. world frame
    PoseSE3 Twj = frame_curr->getPose();
    Rot3 Rwj = Twj.block<3,3>(0,0);
    Rot3 Rjw = Rwj.transpose().eval();
    
    Rot3 dRj = dT.block<3,3>(0,0);
    Pos3 uj  = dT.block<3,1>(0,3);    
    
    Pos3 Rwj_uj = Rwj*uj;

    // std::cout << "framevec.size(): " << framevec.size() << std::endl;
    PoseSE3 Twjm1 = framevec[framevec.size()-2]->getPose();
    Pos3 twjm1 = Twjm1.block<3,1>(0,3);
    
    // Find age > 1 && parallax > 0.5 degrees
    LandmarkPtrVec lms_no_depth;
    LandmarkPtrVec lms_depth;
    for(int m = 0; m < lmvec.size(); ++m){
        const LandmarkPtr& lm = lmvec[m];
        if(    lm->getAge()          >= thres_age_use_ 
            && lm->getLastParallax() >= thres_parallax_use_
            && lm->isTriangulated() == false) 
        {
            lms_no_depth.push_back(lm);        
        }

        if(lm->isTriangulated()) 
            lms_depth.push_back(lm);
    }

    uint32_t M_tmp = lms_no_depth.size();

    // Generate Matrices
    uint32_t mat_size = 3*M_tmp + 1;
    uint32_t idx_end = mat_size - 1;
    SpMat AtA(mat_size, mat_size);
    SpVec Atb(mat_size, 1);
    Atb.coeffRef(idx_end,0) = 0.0f;

    // Initialize AtA matrix with zeros
    AtA.reserve(Eigen::VectorXf::Constant(mat_size, 4)); // 한 행에 4개의 원소밖에 없을거다.
    int idx = 0;
    SpTripletList tplist_tmp;
    tplist_tmp.resize(0);

    uint32_t idx_mat[3]={0,1,2};
    for(int i = 0; i < M_tmp; ++i){
        // block initialization
        tplist_tmp.emplace_back(idx_mat[0],idx_mat[0], 0.0f);
        tplist_tmp.emplace_back(idx_mat[0],idx_mat[1], 0.0f);
        tplist_tmp.emplace_back(idx_mat[0],idx_mat[2], 0.0f);

        tplist_tmp.emplace_back(idx_mat[1],idx_mat[0], 0.0f);
        tplist_tmp.emplace_back(idx_mat[1],idx_mat[1], 0.0f);
        tplist_tmp.emplace_back(idx_mat[1],idx_mat[2], 0.0f);

        tplist_tmp.emplace_back(idx_mat[2],idx_mat[0], 0.0f);
        tplist_tmp.emplace_back(idx_mat[2],idx_mat[1], 0.0f);
        tplist_tmp.emplace_back(idx_mat[2],idx_mat[2], 0.0f);

        tplist_tmp.emplace_back(idx_mat[0], idx_end, 0.0f);
        tplist_tmp.emplace_back(idx_mat[1], idx_end, 0.0f);
        tplist_tmp.emplace_back(idx_mat[2], idx_end, 0.0f);

        tplist_tmp.emplace_back(idx_end, idx_mat[0], 0.0f);
        tplist_tmp.emplace_back(idx_end, idx_mat[1], 0.0f);
        tplist_tmp.emplace_back(idx_end, idx_mat[2], 0.0f);
        
        idx_mat[0] += 3;
        idx_mat[1] += 3;
        idx_mat[2] += 3;
    }
    tplist_tmp.emplace_back(idx_end, idx_end, 0);

    AtA.setFromTriplets(tplist_tmp.begin(),tplist_tmp.end());
    // std::cout << "# elems: " <<tplist_tmp.size() << ", total elems: " << mat_size* mat_size << std::endl;
    // std::cout << "size: " << 9*M_tmp + 6*M_tmp + 1 << "\n";
    // std::cout << "Mat inner size: " << AtA.innerSize() <<", " << AtA.outerSize() << ", total size: "<< mat_size << std::endl;

    // i-th landmark (with no depth)
    Eigen::Matrix<float,3,3> AtAlast_tmp = Eigen::Matrix<float,3,3>::Zero();
    
    Eigen::Matrix<float,2,3> Fik;
    Eigen::Matrix<float,3,3> FiktFik;
    Eigen::Matrix<float,3,1> FiktFik_Rwk_uj;
    Eigen::Matrix<float,3,1> FiktFik_twjm1;
    Eigen::Matrix<float,3,1> FiktFik_twk;
    idx_mat[0] = 0; idx_mat[1] = 1; idx_mat[2] = 2;
    for(int ii = 0; ii < M_tmp; ++ii){
        const LandmarkPtr& lm = lms_no_depth[ii];
        const PixelVec& pts_history = lm->getObservations();

        // 현재 랜드마크와 연관된 가장 오래된 FRAME ID를 추출한다.
        int j_end = lm->getRelatedFramePtr().front()->getID(); 
        if(j - j_end + 1 >= thres_age_past_horizon_)
            j_end = j - thres_age_past_horizon_ + 1;

        if( j == j_end)
            throw std::runtime_error("j == j_end");
        
        // std::cout << "j and j_end : " << j << ", " << j_end << std::endl;
        // k-th frame relatd to i-th landmark
        int idx_p = pts_history.size()-1;
        for(int k = j; k >= j_end; --k){
            const Pixel& p = pts_history[idx_p];
            --idx_p;

            PoseSE3 Twk = framevec[k]->getPose();
            Rot3 Rwk = Twk.block<3,3>(0,0);
            Pos3 twk = Twk.block<3,1>(0,3);

            Point xik;
            xik << fxinv*(p.x - cx), fyinv*(p.y - cy), 1.0f;

            Point xik_p;
            xik_p = Rwk*xik;
            Fik << -xik_p(2), 0.0f, xik_p(0),
                    0.0f,-xik_p(2), xik_p(1);

            FiktFik = Fik.transpose()*Fik;
            if(k == j){
                // AtA (side)
                FiktFik_Rwk_uj = FiktFik*Rwk*uj; // 3x1

                AtA.coeffRef(idx_mat[0], idx_end) = FiktFik_Rwk_uj(0);
                AtA.coeffRef(idx_mat[1], idx_end) = FiktFik_Rwk_uj(1);
                AtA.coeffRef(idx_mat[2], idx_end) = FiktFik_Rwk_uj(2);

                AtA.coeffRef(idx_end, idx_mat[0]) = FiktFik_Rwk_uj(0);
                AtA.coeffRef(idx_end, idx_mat[1]) = FiktFik_Rwk_uj(1);
                AtA.coeffRef(idx_end, idx_mat[2]) = FiktFik_Rwk_uj(2);

                // AtA (last, 3x3)
                AtAlast_tmp += FiktFik;

                // Atb
                FiktFik_twjm1 = FiktFik*twjm1;
                Atb.coeffRef(idx_mat[0], 0) += FiktFik_twjm1(0);
                Atb.coeffRef(idx_mat[1], 0) += FiktFik_twjm1(1);
                Atb.coeffRef(idx_mat[2], 0) += FiktFik_twjm1(2);

                // Atb (end)
                float ujt_Rjw_FiktFik_twjm1 = uj.transpose()*Rjw*FiktFik*twjm1;
                Atb.coeffRef(idx_end,0) += ujt_Rjw_FiktFik_twjm1;
            }
            else{
                // Atb
                FiktFik_twk = FiktFik*twk;
                Atb.coeffRef(idx_mat[0], 0) += FiktFik_twk(0);
                Atb.coeffRef(idx_mat[1], 0) += FiktFik_twk(1);
                Atb.coeffRef(idx_mat[2], 0) += FiktFik_twk(2);
            }
            // AtA (block) (Symmetric, zeros can be ignored.)
            AtA.coeffRef(idx_mat[0], idx_mat[0]) += FiktFik(0,0);
            AtA.coeffRef(idx_mat[0], idx_mat[1]) += FiktFik(0,1);
            AtA.coeffRef(idx_mat[0], idx_mat[2]) += FiktFik(0,2);

            AtA.coeffRef(idx_mat[1], idx_mat[0]) += FiktFik(1,0);
            AtA.coeffRef(idx_mat[1], idx_mat[1]) += FiktFik(1,1);
            AtA.coeffRef(idx_mat[1], idx_mat[2]) += FiktFik(1,2);

            AtA.coeffRef(idx_mat[2], idx_mat[0]) += FiktFik(2,0);
            AtA.coeffRef(idx_mat[2], idx_mat[1]) += FiktFik(2,1);
            AtA.coeffRef(idx_mat[2], idx_mat[2]) += FiktFik(2,2);
        } // END k

        idx_mat[0] += 3; idx_mat[1] += 3; idx_mat[2] += 3;
    } // END ii

    // i-th landmark with depth
    for(int ii = 0; ii < lms_depth.size(); ++ii){
        const LandmarkPtr& lm = lms_depth[ii];

        Pixel pj  = lm->getObservations().back();
        Point Xwi = lm->get3DPoint();

        Eigen::Matrix<float,3,1> xij;
        xij << fxinv*(pj.x-cx), fyinv*(pj.y-cy), 1.0f;

        Eigen::Matrix<float,3,1> xij_p = Rwj*xij;
        Eigen::Matrix<float,2,3> Fij;
        Fij << -xij_p(2), 0.0f, xij_p(0),
               0.0f, -xij_p(2), xij_p(1);
        Eigen::Matrix<float,3,3> FijtFij = Fij.transpose()*Fij;
        AtAlast_tmp += FijtFij;

        float ujt_Rjw_Rij_FijtFij_twjm1_Xwi = uj.transpose()*Rjw*FijtFij*(twjm1-Xwi);
        Atb.coeffRef(idx_end,0) += ujt_Rjw_Rij_FijtFij_twjm1_Xwi;
    }

    // AtA (last)
    AtA.coeffRef(idx_end, idx_end) = uj.transpose()*Rjw*AtAlast_tmp*Rwj*uj;

    // Efficiently solve theta = AtA^-1* Atb;
    // std::cout << "Solve Theta...\n";
    SpVec theta;
    this->solveLeastSquares_SFP(AtA, Atb, M_tmp, theta);

    // Update motion
    float scale_est = theta.coeff(3*M_tmp);
    if(!isnan(scale_est)){
        // if(scale_est <0 ) scale_est = -scale_est;
        PoseSE3 dT10_est;
        dT10_est << dRj, uj*scale_est,0,0,0,1;
        frame_curr->setPose(Twjm1*dT10_est.inverse());
        frame_curr->setPoseDiff10(dT10_est);
        frame_curr->setScale(scale_est);


        // Update points
        idx = 0;
        for(int ii = 0; ii < M_tmp; ++ii){
            Point Xw;
            Xw << theta.coeff(idx), theta.coeff(++idx), theta.coeff(++idx);
            ++idx;
            
            const LandmarkPtr& lm = lms_no_depth[ii];
            if(lm->getAge() >= thres_age_recon_ 
             && lm->getLastParallax() >= thres_parallax_recon_)
            {
                lm->set3DPoint(Xw);
            }
        } 
    }
    

    // std::cout << " Solve OK!\n";

};
