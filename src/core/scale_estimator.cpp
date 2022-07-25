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


    // SFP parameters
    N_max_past_ = 5;

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

void ScaleEstimator::module_ScaleForwardPropagation(const LandmarkPtrVec& lmvec, const FramePtrVec& framevec, const PoseSE3& dT10)
{
    // intrinsic parameters
    float fx = cam_->fx();
    float fy = cam_->fy();
    float fxinv = cam_->fxinv();
    float fyinv = cam_->fyinv();
    float cx = cam_->cx();
    float cy = cam_->cy();
    Eigen::Matrix3f K    = cam_->K();
    Eigen::Matrix3f Kinv = cam_->Kinv();

    // Pixels tracked in this frame
    lmvec;

    // Get relative motion
    PoseSE3 dT = dT10;
    
    // Get rotation of this frame w.r.t. world frame
    PoseSE3 Twj = framevec.back()->getPose();
    Rot3 Rwj = Twj.block<3,3>(0,0);
    Rot3 Rjw = Rwj.transpose();
    
    Rot3 dRj = dT.block<3,3>(0,0);
    Pos3 uj  = dT.block<3,1>(0,3);

    FramePtr frame_curr = framevec.back();
    int j = frame_curr->getID() ;

    PoseSE3 Twjm1 = framevec[framevec.size()-2]->getPose();
    Pos3 twjm1 = Twjm1.block<3,1>(0,3);
    
    // Find age > 1 && parallax > 0.5 degrees
    LandmarkPtrVec lms_no_depth;
    LandmarkPtrVec lms_depth;
    
    for(int m = 0; m < lmvec.size(); ++m){
        const LandmarkPtr& lm = lmvec[m];
        if(lm->getAge() > 1 && lm->getTriangulated() == false) {
            lms_no_depth.push_back(lm);    
        }
        if(lm->getTriangulated()){
            lms_depth.push_back(lm);
        }
    }

    uint32_t M_tmp = lms_no_depth.size();
    std::cout << "No depth size: " << M_tmp << std::endl;
    std::cout << frame_curr->getID() <<" -th frame, Mtmp : " << M_tmp << std::endl;

    // Generate Matrices
    uint32_t mat_size = 3*M_tmp + 1;
    SpMat AtA(3*M_tmp+1, 3*M_tmp+1);
    SpVec Atb(3*M_tmp+1, 1); Atb.coeffRef(mat_size,0) = 0.0f;
    SpTripletList Tplist;
    AtA.reserve(Eigen::VectorXi::Constant(3*M_tmp+1, 20)); // 한 행에 4개의 원소밖에 없을거다.

    // i-th landmark (with no depth)
    for(int ii = 0; ii < M_tmp; ++ii){
        int idx_mat[2] = {3*ii, 3*ii+2};

        PixelVec pts_history = lmvec[ii]->getObservations();

        int j_end = lmvec[ii]->getRelatedFramePtr().front()->getID();
        if(j - j_end + 1 >= N_max_past_){
            j_end = j - N_max_past_ + 1;
        }

        if( j == j_end){
            throw std::runtime_error("j == j_end");
        }

        // k-th frame relatd to i-th landmark
        for(int k = j; j >= j_end; --j){
            PoseSE3 Twk = framevec[j]->getPose();
            Rot3 Rwk = Twk.block<3,3>(0,0);
            Pos3 twk = Twk.block<3,1>(0,3);

            Point xik;
            xik << fxinv*(pts_history[idx_p].x - cx), fyinv*(pts_history[idx_p].y - cy), 1.0f;

            Point xik_p;
            xik_p = Rwk*xik;
            Eigen::Matrix<float,2,3> Fik;
            Fik << -xik_p(2), 0.0f, xik_p(0),
                    0.0f,-xik_p(2), xik_p(1);

            Eigen::Matrix<float,3,3> FiktFik = Fik.transpose()*Fik;
            Eigen::Matrix<float,3,3> AtAlast_tmp = Eigen::Matrix<float,3,3>::Zero();
            if(k == j){
                // AtA (side)
                Eigen::Matrix<float,3,1> FiktFik_Rwk_uj = FiktFik*Rwk*uj;

                this->fillTriplet(Tplist, 
                    idx_mat[0], idx_mat[1], 
                    mat_size - 1, mat_size - 1, FiktFik_Rwk_uj);
                this->fillTriplet(Tplist, 
                    mat_size - 1, mat_size - 1, 
                    idx_mat[0], idx_mat[1], FiktFik_Rwk_uj.transpose());

                // AtA (last, 3x3)
                AtAlast_tmp += FiktFik;

                // Atb
                Eigen::Matrix<float,3,1> FiktFik_twjm1 = FiktFik*twjm1;
                Atb.coeffRef(idx_mat[0],   0) += FiktFik_twjm1(0);
                Atb.coeffRef(idx_mat[0]+1, 0) += FiktFik_twjm1(1);
                Atb.coeffRef(idx_mat[0]+2, 0) += FiktFik_twjm1(2);

                // Atb (end)
                float ujt_Rjw_FiktFik_twjm1 = uj.transpose()*Rjw*FiktFik*twjm1;
                Atb.coeffRef(mat_size,0) += ujt_Rjw_FiktFik_twjm1;
            }
            else{
                // Atb
                Eigen::Matrix<float,3,1> FiktFik_twk = FiktFik*twk;
                Atb.coeffRef(idx_mat[0],   0) += FiktFik_twk(0);
                Atb.coeffRef(idx_mat[0]+1, 0) += FiktFik_twk(1);
                Atb.coeffRef(idx_mat[0]+2, 0) += FiktFik_twk(2);
            }
            // AtA (block)
            this->addTriplet(Tplist, 
                    idx_mat[0], idx_mat[1], 
                    idx_mat[0], idx_mat[1], FiktFik_Rwk_uj);
            AtA(idx_mat,idx_mat) = AtA(idx_mat,idx_mat) + FiktFik;

        }
    }

    // i-th landmark with depth
    for(int ii = 0; ii < lms_depth.size(); ++ii){

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



inline void ScaleEstimator::fillTriplet(SpTripletList& Tri,  
    const int& idx_row0, const int& idx_row1,
    const int& idx_col0, const int& idx_col1, const Eigen::MatrixXf& mat)
{
    int dim_hori = idx_col1 - idx_col0 + 1;
    int dim_vert = idx_row1 - idx_row0 + 1;

    if(mat.cols() != dim_hori) throw std::runtime_error("ScaleEstimator::fillTriplet(...), mat.cols() != dim_hori\n");
    if(mat.rows() != dim_vert) throw std::runtime_error("ScaleEstimator::fillTriplet(...), mat.rows() != dim_vert\n");

    for(int u = 0; u < dim_hori; ++u) {
        for(int v = 0; v < dim_vert; ++v) {
            Tri.push_back(SpTriplet(v + idx_row0, u + idx_col0, mat(v,u)));
        }
    }
};
    

inline void addTriplet(SpTripletList& Tri,  
    const int& idx_row0, const int& idx_row1,
    const int& idx_col0, const int& idx_col1, const Eigen::MatrixXf& mat)
{
    int dim_hori = idx_col1 - idx_col0 + 1;
    int dim_vert = idx_row1 - idx_row0 + 1;

    if(mat.cols() != dim_hori) throw std::runtime_error("ScaleEstimator::addTriplet(...), mat.cols() != dim_hori\n");
    if(mat.rows() != dim_vert) throw std::runtime_error("ScaleEstimator::addTriplet(...), mat.rows() != dim_vert\n");

    for(int u = 0; u < dim_hori; ++u) {
        for(int v = 0; v < dim_vert; ++v) {
            Tri.push_back(SpTriplet(v + idx_row0, u + idx_col0, mat(v,u)));
        }
    }
}
