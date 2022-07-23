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

    printf(" - ScaleEstimator is constructed.\n");
};  
ScaleEstimator::~ScaleEstimator(){
    // Terminate signal .
    std::cerr << "ScaleEstimator - terminate signal published...\n";
    terminate_promise_.set_value();
    
    mut_->lock();
    *flag_do_ASR_ = true;
    mut_->unlock();
    cond_var_->notify_all();

    // wait for TX & RX threads to terminate ...
    std::cerr << "                   - waiting 1 second to join TX / RX threads ...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if(thread_.joinable()) thread_.join();
    std::cerr << "                   - scale estimator thread joins successfully.\n";

    printf(" - ScaleEstimator is destructed.\n");
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
    std::cerr << "ScaleEstimator - thread receives termination signal.\n";
};