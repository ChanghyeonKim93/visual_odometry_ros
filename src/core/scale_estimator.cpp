#include "core/scale_estimator.h"

std::shared_ptr<Camera> ScaleEstimator::cam_ = nullptr;

ScaleEstimator::ScaleEstimator(const std::shared_ptr<std::mutex> mut, 
    const std::shared_ptr<std::condition_variable> cond_var,
    const std::shared_ptr<bool> flag_do_ASR){
    // Mutex from the outside
    mut_ = mut;
    cond_var_ = cond_var;
    flag_do_ASR_ = flag_do_ASR;

    // Detecting turn region variables
    cnt_turn_ = 0;

    thres_cnt_turn_ = 15;
    // thres_psi_ = 1.0 * M_PI / 180.0;
    thres_psi_ = 0.02;

    terminate_future_ = terminate_promise_.get_future();
    runThread();

    printf(" - SCALE_ESTIMATOR is constructed.\n");
};  

ScaleEstimator::~ScaleEstimator(){
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

    if(thread_.joinable()) thread_.join();
    std::cerr << "                   - SCALE_ESTIMATOR thread joins successfully.\n";

    printf(" - SCALE_ESTIMATOR is destructed.\n");
};


void ScaleEstimator::runThread(){
    thread_ = std::thread([&](){ process(terminate_future_); } );
};

void ScaleEstimator::process(std::shared_future<void> terminate_signal){
    while(true){
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

void ScaleEstimator::module_ScaleForwardPropagation(const LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3 dT10)
{
    // Pixels tracked in this frame

    // Get relative motion
    // dT10

    // Get rotation of this frame w.r.t. world frame
    // Rwj
    // Rjw = Rwj.transpose();
    
    // dRj = dT10(1:3,1:3);
    // uj = dT10(1:3,4);
    
    // Find age > 1 && parallax > 0.5 degrees
    // lms_nodepth
    // lms_depth
    
    for(int m = 0; m < lmvec.size(); ++m){

        if(lmvec[m]->getAge() > 1) {
            
        }
    }
    

};


bool ScaleEstimator::detectTurnRegions(const FramePtr& frame){
    bool flag_turn_detected = false;

    float psi = frame->getSteeringAngle();
    std::cout <<"psi : " << psi << ", threspsi: " << thres_psi_ << std::endl;
    if( abs(psi) > thres_psi_ ) { // Current psi is over the threshold
        frames_t1_.push_back(frame); // Stack the 
        ++cnt_turn_;
    }
    else { // end of array of turn regions
        if(cnt_turn_ >= thres_cnt_turn_){ // sufficient frames
            // Do Scale Forward Propagation
            frames_t0_; // Ft0
            frames_t1_; // Ft1
            frames_u_; // Fu

            std::cout << " TURN REGION IS DETECTED!\n";
            
            flag_turn_detected = true;

            // Update turn regions
            frames_t0_ = frames_t1_;
            std::cout << "TURN regions:\n";
            for(int i = 0; i < frames_t0_.size(); ++i){
                std::cout << frames_t0_[i]->getID() << " " ;
                frames_all_t_.push_back(frames_t0_[i]);
            }
            std::cout << "\n";
        }
        else { // insufficient frames. The stacked frames are not of the turning region.
            for(auto f : frames_t1_)
                frames_u_.push_back(f);
        }
        frames_t1_.resize(0);
        cnt_turn_ = 0;
    }

    return flag_turn_detected;
};

const FramePtrVec& ScaleEstimator::getAllTurnRegions() const{
    return frames_all_t_;
};