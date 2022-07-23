#include "core/scale_estimator.h"

std::shared_ptr<Camera> ScaleEstimator::cam_ = nullptr;

ScaleEstimator::ScaleEstimator(const std::shared_ptr<std::mutex> mut, 
    const std::shared_ptr<std::condition_variable> cond_var,
    const std::shared_ptr<bool> flag_do_ASR){
    // Mutex from the outside
    mut_ = mut;
    cond_var_ = cond_var;
    flag_do_ASR_ = flag_do_ASR;

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
        
        std::cout << "THREAD GETS CONDITION VARIAVLE!\n";

        *flag_do_ASR_ = false;
        lk.unlock();

        std::future_status terminate_status = terminate_signal.wait_for(std::chrono::microseconds(1000));
        if (terminate_status == std::future_status::ready) {
                    

            break;
        }
    }
    std::cerr << "SCALE_ESTIMATOR - thread receives termination signal.\n";
};


void ScaleEstimator::module_ScaleForwardPropagation(LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3 dT10)
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
            std::cout<<"ok\n";
        }
    }
    

};