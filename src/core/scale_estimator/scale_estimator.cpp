#include "core/scale_estimator/scale_estimator.h"

ScaleEstimator::ScaleEstimator(
    const std::shared_ptr<Camera> cam,
    const float& L,
    const std::shared_ptr<std::mutex> mut, 
    const std::shared_ptr<std::condition_variable> cond_var,
    const std::shared_ptr<bool> flag_do_ASR)
: frame_prev_(nullptr), L_(L), cnt_turn_(0)
{
    // Setting camera
    cam_ = cam;

    // Generate SFP, ASR
    sfp_module_ = std::make_shared<ScaleForwardPropagation>(cam);
    asr_module_ = std::make_shared<AbsoluteScaleRecovery>(cam);

    // Mutex from the outside
    mut_         = mut;
    cond_var_    = cond_var;
    flag_do_ASR_ = flag_do_ASR;

    // Detecting turn region variables
    THRES_CNT_TURN_ = 15;
    THRES_PSI_      = 2.0*M_PI/180.0;


    // Run process thread.
    terminate_future_ = terminate_promise_.get_future();
    this->runThread();

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
        if (terminate_status == std::future_status::ready) 
        {
            break;
        }
    }

    std::cerr << "SCALE_ESTIMATOR - thread receives termination signal.\n";
};


void ScaleEstimator::setTurnRegion_ThresPsi(float psi){
    THRES_PSI_ = psi;
};
void ScaleEstimator::setTurnRegion_ThresCountTurn(uint32_t thres_cnt_turn){
    THRES_CNT_TURN_ = thres_cnt_turn;
};


void ScaleEstimator::insertNewFrame(const FramePtr& frame)
{
    // Insert the frame.
    frames_all_.push_back(frame);
    
    // Check whether this is a turning frame or not
    bool flag_turn_detected = this->detectTurnRegions(frame);

    // flag is not on 
    
    // if(flag_turn_detected)
    // {
    //     // This is a turning frame
    // }
    // else
    // {
    //     // This is not a turning frame.
    // }

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
        flag_turn_detected = false;

        return flag_turn_detected;
    }

    // Calculate the steering angle from the previous frame.
    const PoseSE3& Twp = this->frame_prev_->getPose();
    const PoseSE3& Twc = frame->getPose();

    PoseSE3 Tpc = Twp.inverse()*Twc; 

    const Rot3& Rpc = Tpc.block<3,3>(0,0);
    const Pos3& tpc = Tpc.block<3,1>(0,3);

    float psi_pc = this->calcSteeringAngle(Rpc);

    std::cout << frame->getID() <<"-th frame is keyframe? : " << (frame->isKeyframe() ? "YES" : "NO") << std::endl;
    std::cout << frame->getID() << "-th steering angle: " << psi_pc*R2D << " [deg]\n";
    
    // 현재 구한 steering angle이 문턱값보다 높으면, turning frame이다.
    if( abs(psi_pc) > this->THRES_PSI_)
    {
        // This is a turning frame.
        // Calculate scale raw
        float scale_raw = this->calcScale(psi_pc, tpc, this->L_);

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
        /* 문턱값보다 낮으면 두 가지 경우 : 
            1) cnt_turn > 0
                if) cnt_turn >= THRES_CNT_TURN_
                    --> DO ASR
                    --> prev = curr
                    --> unconstrained.resize(0)
                    --> frames_turn_curr.resize(0)
                else)
                    --> frames_turn_curr 을 unconstrained로 보낸다.
                    --> frames_turn_curr.resize(0)
                
                --> cnt_turn = 0 ;

            2) 그냥 직진이다.
                --> 현재 프레임을 unconstrained 로 보낸다.
        */
        if(cnt_turn_ > 0)
        {
            if(cnt_turn_ >= THRES_CNT_TURN_)
            {
                // New Turning Region is detected.
                // Sufficient frames, do Absolute Scale Recovery (ASR)
                std::cout << " ======================                                =====================\n";
                std::cout << " ======================                                =====================\n";
                std::cout << " ====================== A NEW TURN REGION IS DETECTED! =====================\n";
                std::cout << " ======================                                =====================\n";
                std::cout << " ======================                                =====================\n";

                // Calculate refined scales of 'frames_turn_curr_'
                int n_Fcurr = frames_turn_curr_.size();
                std::vector<float> trans_norm_t1; // translation norm
                std::vector<float> ratios_t1; // ratio ( ==  scale_raw / trans.norm() )
                for(const FramePtr& f : frames_turn_curr_)
                {
                    // Pos3 t01 = f->
                    const PoseSE3& Twp = f->getPreviousTurningFrame()->getPose();
                    const PoseSE3& Twc = f->getPose();
                    PoseSE3 Tpc = Twp.inverse()*Twc;

                    const Pos3& tpc = Tpc.block<3,1>(0,3);

                    // Scale vs. translation norm.
                    float trans_norm = tpc.norm();
                    float scale_raw  = f->getScaleRaw();
                    trans_norm_t1.push_back(trans_norm);
                    ratios_t1.push_back(scale_raw/trans_norm);
                }

                // Get median ratio
                float norm_scaler = 0.0;
                std::sort(ratios_t1.begin(), ratios_t1.end());
                norm_scaler = ratios_t1[(int)(ratios_t1.size()*0.5)];

                std::cout << "    - norm_scaler : " << norm_scaler << std::endl;

                // Set refine scale information.
                for(int j = 0; j < n_Fcurr; ++j)
                {
                    const FramePtr& f = frames_turn_curr_.at(j);
                    float scale_refined = trans_norm_t1.at(j) * norm_scaler;
                    f->setScale(scale_refined);
                }

                // Do Absolute Scale Recovery (ASR)
                std::cout << "    - # frames- Ft0, Fu, Ft1: "
                    << frames_turn_prev_.size() <<", "
                    << frames_unconstrained_.size() << ", "
                    << frames_turn_curr_.size() << std::endl;

                std::cout <<  "    - TURN prev: ";
                for(auto& f : frames_turn_prev_)
                    std::cout << f->getID() << " ";
                std::cout << std::endl;

                std::cout << "    - Unconstrained: ";
                for(auto& f : frames_unconstrained_)
                    std::cout << f->getID() << " ";
                std::cout << std::endl;
                
                std::cout << "    - TURN curr: ";
                for(auto& f : frames_turn_curr_)
                    std::cout << f->getID() << " ";
                std::cout << std::endl;

                // If there is a previous turning frames, do ASR
                if( frames_turn_prev_.size() > 0)
                {
                    // Run the Absolute Scale Recovery (ASR) Module.
                    asr_module_->runASR(
                        frames_turn_prev_,
                        frames_unconstrained_,
                        frames_turn_curr_);
                }
                                
                // Update previous turning region & empty the unconstrained region
                frames_unconstrained_.resize(0);
                frames_turn_prev_.resize(frames_turn_curr_.size());
                std::copy(frames_turn_curr_.begin(), frames_turn_curr_.end(), frames_turn_prev_.begin());
                
                frames_turn_curr_.resize(0);
                cnt_turn_ = 0;
            }
            else
            {
                // Not Turning Region.
                // Insufficient frames. The stacked frames are not of turning region.
                // Make them to unconstrained frames.
                for(auto f : frames_turn_curr_)
                {
                    f->cancelThisTurningFrame();
                    frames_unconstrained_.push_back(f);
                }
               
                frames_turn_curr_.resize(0);
                cnt_turn_ = 0;
            }
        }
        else
        {
            frames_unconstrained_.push_back(frame);
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

float ScaleEstimator::calcScale(float psi, const Pos3& t01, float L){
    
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

float ScaleEstimator::calcSteeringAngle(const Rot3& R)
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
        throw std::runtime_error("MotionEstimator-'calcSteeringAngle', std::isnan(psi) == true");    
    
    if( vjdot < 0 )
        psi = -psi;

    psi = psi*(vjdot);
    std::cout << "psi: " << psi*R2D << " [deg]\n";

    return psi;
};






const FramePtrVec& ScaleEstimator::getAllTurnRegions() const{
    return frames_all_turn_;
};

const FramePtrVec& ScaleEstimator::getLastTurnRegion() const{
    return frames_turn_prev_;
};

