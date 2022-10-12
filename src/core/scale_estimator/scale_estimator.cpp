#include "core/scale_estimator/scale_estimator.h"

ScaleEstimator::ScaleEstimator(
    const std::shared_ptr<Camera> cam,
    const float& L,
    const std::shared_ptr<std::mutex> mut, 
    const std::shared_ptr<std::condition_variable> cond_var,
    const std::shared_ptr<bool> flag_do_ASR)
: frame_prev_(nullptr), L_(L)
{

    cam_ = cam;

    // Mutex from the outside
    mut_         = mut;
    cond_var_    = cond_var;
    flag_do_ASR_ = flag_do_ASR;

    // Detecting turn region variables
    cnt_turn_    = 0;

    THRES_CNT_TURN_ = 15;
    // THRES_PSI_ = 1.0 * M_PI / 180.0;
    THRES_PSI_   = 2.0*M_PI/180.0;

    // SFP parameters
    thres_age_past_horizon_ = 20;
    thres_age_use_          = 2;
    thres_age_recon_        = 15;
    thres_flow_             = 5.0;

    thres_parallax_use_     = 0.5*D2R;
    thres_parallax_recon_   = 60*D2R;

    terminate_future_ = terminate_promise_.get_future();
    runThread();

    printf(" - SCALE_ESTIMATOR is constructed.\n");
};


ScaleEstimator::~ScaleEstimator()
{
    // Terminate signal .
    std::cerr << "SCALE_ESTIMATOR - terminate signal published...\n";
    terminate_promise_.set_value();

    // Notify the thread to run the while loop.
    mut_->lock();
    *flag_do_ASR_ = true;
    mut_->unlock();
    
    cond_var_->notify_all();

    // wait for TX & RX threads to terminate ...
    std::cerr << "                   - waiting 1 second to join a thread ...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if(thread_.joinable()) 
    {
        thread_.join();
    }

    std::cerr << "                   - SCALE_ESTIMATOR thread joins successfully.\n";

    printf(" - SCALE_ESTIMATOR is destructed.\n");
};


void ScaleEstimator::runThread()
{
    thread_ = std::thread([&](){ process(terminate_future_); } );
};


void ScaleEstimator::process(std::shared_future<void> terminate_signal)
{
    while(true)
    {
        std::unique_lock<std::mutex> lk(*mut_);

        cond_var_->wait(lk, [=] { return (*flag_do_ASR_); });
        
        *flag_do_ASR_ = false;
        lk.unlock();

        std::future_status terminate_status = terminate_signal.wait_for(std::chrono::microseconds(1000));
        if (terminate_status == std::future_status::ready) {
                    

            break;
        }
    }
    std::cerr << "SCALE_ESTIMATOR - thread receives termination signal.\n";
};


void ScaleEstimator::insertNewFrame(const FramePtr& frame)
{
    // Insert the frame.
    frames_all_.push_back(frame);

    // If the last frame is not nullptr, try to detect turning region
    if(frame_prev_ != nullptr)
    {

    }

    // Change the previous frame.
    frame_prev_ = frame;

};


bool ScaleEstimator::detectTurnRegions(const FramePtr& frame)
{
    bool flag_turn_detected = false;

    // 만약 previous frame이 없으면, 데이터 넣어주고 리턴. 
    if(this->frame_prev_ == nullptr) 
    {
        this->frame_prev_ = frame;
        flag_turn_detected = true;

        return flag_turn_detected;
    }

    // Calculate the steering angle from the previous frame.
    const PoseSE3& Twp = this->frame_prev_->getPose();
    const PoseSE3& Twc = frame->getPose();

    PoseSE3 Tpc = Twp.inverse()*Twc; 

    const Rot3& Rpc = Tpc.block<3,3>(0,0);
    const Pos3& tpc = Tpc.block<3,1>(0,3);

    float psi_pc = this->calcSteeringAngleFromRotationMat(Rpc);

    std::cout << frame->getID() <<"-th frame is keyframe? : " << (frame->isKeyframe() ? "YES" : "NO") << std::endl;
    std::cout << frame->getID() << "-th steering angle: " << psi_pc*R2D << " [deg]\n";
    
    // 현재 구한 steering angle이 문턱값보다 높으면, turning frame이다.
    if( abs(psi_pc) > this->THRES_PSI_)
    {
        // This is a turning frame.
        // Calculate scale raw
        float scale_raw = this->calcScaleByKinematics(psi_pc, tpc, this->L_);

        frame->makeThisTurningFrame(frame_prev_);
        frame->setSteeringAngle(psi_pc);        
        frame->setScaleRaw(scale_raw);
        // frame->setScale(); // It can be calculated.
        
        frames_turn_curr_.push_back(frame);        

        // Get T01 
        ++cnt_turn_;
    }
    else
    {
        // End of array of turning frames. 

        // If Sufficient frames, make new turning region.
        if(cnt_turn_ >= THRES_CNT_TURN_)
        {
            // Sufficient frames, 
            // Do Absolute Scale Recovery
            frames_turn_prev_; // F_tp (previous turning region)
            frames_turn_curr_; // F_tc (current turning region)
            frames_unconstrained_; // Fu

            std::cout << " === A NEW TURN REGION IS DETECTED!\n";

            // Calculate refined scales of 'frames_turn_curr_'
            float mean_scale_raw = 0.0f;
            std::vector<float> trans_norm_t1;
            for(auto f : frames_turn_curr_)
            {
                // Pos3 t01 = f->
            }

        }
        else
        { 
            // Insufficient frames. The stacked frames are not of turning region.
            // Make them to unconstrained frames.
            for(auto f : frames_turn_curr_) 
                frames_unconstrained_.push_back(f);
        }
    }

    return flag_turn_detected;
};


// bool ScaleEstimator::detectTurnRegions(
//     const std::shared_ptr<Keyframes>& keyframes, const FramePtr& frame)
// {
//     // 새로운 turning region이 감지되었는지 판단.
//     bool flag_turn_detected = false;

//     // Calculate steering angle from the last keyframe.
//     const PoseSE3& Twk = keyframes->getList().back()->getPose();
//     const PoseSE3& Twc = frame->getPose();
//     PoseSE3 Tkc = Twk.inverse()*Twc; 
        
//     const Rot3& Rkc = Tkc.block<3,3>(0,0);
//     const Pos3& tkc = Tkc.block<3,1>(0,3);

//     float psi_kc = this->calcSteeringAngleFromRotationMat(Rkc);
    
//     frame->setSteeringAngle(psi_kc);
//     frame->setPoseDiffFromLastKeyframe(Tkc);

//     std::cout << frame->getID() << "-th steering angle: " << psi_kc*R2D << " [deg]\n";
        
//     if( abs(psi_kc) > this->THRES_PSI_ )
//     { 
//         // Current psi is over the threshold
//         frames_turn_curr_.push_back(frame); // Stack the 
//         ++cnt_turn_;
//     }
//     else
//     { 
//         // End of array of turn regions
//         if(cnt_turn_ >= THRES_CNT_TURN_)
//         {
//             // Sufficient frames, 
//             // Do Absolute Scale Recovery
//             frames_turn_prev_; // Ft0
//             frames_turn_curr_; // Ft1
//             frames_unconstrained_; // Fu

//             std::cout << " TURN REGION IS DETECTED!\n";

//             // Calculate scale of the Turn regions.
//             float L = this->L_;
//             float mean_scale = 0.0f;
//             std::vector<float> scales_t1;
//             std::vector<float> ratios;
//             for(auto f : frames_turn_curr_)
//             {
//                 Pos3 t01 = f->getPoseDiffFromLastKeyframe().block<3,1>(0,3);
//                 float s = calcScaleByKinematics(f->getSteeringAngle(), t01, L);
                
//                 float ratio = s/t01.norm();

//                 scales_t1.push_back(s);
//                 // PoseSE3 dT10_est;
//                 // dT10_est << dRj, uj*scale_est,0,0,0,1;
//                 // frame_curr->setPose(Twjm1*dT10_est.inverse());
//                 // frame_curr->setPoseDiff10(dT10_est);
//                 std::cout << " ------------------- " 
//                 << f->getID() << "-th image scale : " << s 
//                 << ", est: " << t01.norm() << " ratio: " << ratio 
//                 << ", angle: " << f->getSteeringAngle()*R2D << " [deg]" << std::endl;
                
//                 ratios.push_back(ratio);
//             }

//             std::sort(ratios.begin(), ratios.end());
//             std::cout <<" ratios: ";
//             for(auto r : ratios){
//                 std::cout << r <<" ";
//             }
//             std::cout << std::endl;
//             std::cout <<"ratio median: " << ratios[(int)((float)ratios.size()*0.5f)-1] << std::endl;

//             std::sort(scales_t1.begin(), scales_t1.end());
//             int idx_median = (int)((float)scales_t1.size()*0.5f)-1;
            
//             float scale_turn_median = scales_t1[idx_median];
//             float scale_turn_mean = (scales_t1[idx_median-1]+scales_t1[idx_median]+scales_t1[idx_median+1])*0.33333f;
//             std::cout << "turning scale median : " << scale_turn_median << std::endl;
//             std::cout << "turning scale mean : " << scale_turn_mean << std::endl;
//             for(auto f : frames_turn_curr_){
//                 f->setScale(scale_turn_mean);
//                 f->makeThisTurningFrame();
//             }

//             flag_turn_detected = true;


//             // Update turn regions
//             frames_turn_prev_ = frames_turn_curr_;
//             std::cout << "TURN regions:\n";
//             for(int i = 0; i < frames_turn_prev_.size(); ++i){
//                 std::cout << frames_turn_prev_[i]->getID() << " " ;
//                 frames_all_turn_.push_back(frames_turn_prev_[i]);
//             }
//             std::cout << "\n";
//         }
//         else 
//         { 
//             // insufficient frames. The stacked frames are not of the turning region.
//             for(auto f : frames_turn_curr_)
//                 frames_unconstrained_.push_back(f);
//         }
//         frames_turn_curr_.resize(0);
//         cnt_turn_ = 0;
//     }

//     return flag_turn_detected;
// };








const FramePtrVec& ScaleEstimator::getAllTurnRegions() const{
    return frames_all_turn_;
};

const FramePtrVec& ScaleEstimator::getLastTurnRegion() const{
    return frames_turn_prev_;
};


void ScaleEstimator::solveLeastSquares_SFP(const SpMat& AtA, const SpVec& Atb, uint32_t M_tmp,
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

void ScaleEstimator::calcAinvVec_SFP(const SpMat& AA, std::vector<Mat33>& Ainv_vec, uint32_t M_tmp){
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

void ScaleEstimator::calcAinvB_SFP(const std::vector<Mat33>& Ainv_vec, const SpVec& B, uint32_t M_tmp,
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


void ScaleEstimator::setTurnRegion_ThresPsi(float psi){
    THRES_PSI_ = psi;
};
void ScaleEstimator::setTurnRegion_ThresCountTurn(uint32_t thres_cnt_turn){
    THRES_CNT_TURN_ = thres_cnt_turn;
};
void ScaleEstimator::setSFP_ThresAgePastHorizon(uint32_t age_past_horizon){
    thres_age_past_horizon_ = age_past_horizon;
};
void ScaleEstimator::setSFP_ThresAgeUse(uint32_t age_use){
    thres_age_use_ = age_use;
};
void ScaleEstimator::setSFP_ThresAgeRecon(uint32_t age_recon){
    thres_age_recon_ = age_recon;
};
void ScaleEstimator::setSFP_ThresParallaxUse(float thres_parallax_use){
    thres_parallax_use_ = thres_parallax_use;
};
void ScaleEstimator::setSFP_ThresParallaxRecon(float thres_parallax_recon){
    thres_parallax_recon_ = thres_parallax_recon;
};

float ScaleEstimator::calcScaleByKinematics(float psi, const Pos3& t01, float L){
    
    // projection 
    Vec3 j_vec(0,1,0);
    Vec3 tp = t01 - (t01.dot(j_vec))*j_vec;
    
    float costheta = t01(2)/t01.norm();
    if(costheta >= 0.99999f) costheta = 0.99999f;
    if(costheta <= -0.99999f) costheta = -0.99999f;


    float costheta_p = tp(2)/tp.norm();
    if(costheta_p >= 0.99999f)  costheta_p = 0.99999f;
    if(costheta_p <= -0.99999f) costheta_p = -0.99999f;


    float theta = acos(costheta);
    float theta_p = acos(costheta_p);
    std::cout << "theta: " << theta*R2D <<" [deg], theta_p: " << theta_p*R2D << " [deg]\n";


    psi   = abs(psi);
    theta = abs(theta_p);

    float s = L*2.0f*(sin(psi)/(sin(theta)-sin(psi-theta)));
    return s;
};

float ScaleEstimator::calcSteeringAngleFromRotationMat(const Rot3& R)
{
    // Mat33 S = R-R.transpose();
    float inCos = 0.5f*(R.trace()-1.0f);
    if(inCos >=  0.999999999) 
        inCos =  0.999999999;

    if(inCos <= -0.999999999)
        inCos = -0.999999999;

    float psi = acos(inCos);

    Vec3 v; // rotation vector direction.
    v << R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1);
    v = v/v.norm();
    std::cout << "v: " << v.transpose() << std::endl;

    Vec3 j_vec(0,1,0);
    float vjdot = v.dot(j_vec);
    std::cout << "vjdot: " << vjdot << std::endl;

    std::cout <<"psi : " << psi*R2D << " [deg], psi_p: " << psi*(vjdot)*R2D << " [deg]\n";

    if(std::isnan(psi))
        throw std::runtime_error("MotionEstimator-'calcSteeringAngleFromRotationMat', std::isnan(psi) == true");    
    
    if( vjdot < 0 )
        psi = -psi;

    psi = psi*(vjdot);
    std::cout << "psi: " << psi*R2D << " [deg]\n";

    return psi;
};



void ScaleEstimator::module_ScaleForwardPropagation(const LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3& dT10)
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
