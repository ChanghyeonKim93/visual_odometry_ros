#include "core/scale_estimator/absolute_scale_recovery.h"

AbsoluteScaleRecovery::AbsoluteScaleRecovery(const std::shared_ptr<Camera>& cam)
: cam_(cam)
{
    // Make SPQ solver
    sqp_solver_ = std::make_shared<SparseBundleAdjustmentScaleSQPSolver>();

    // OK!
    std::cerr << "AbsoluteScaleRecovery is constructed.\n";
};


AbsoluteScaleRecovery::~AbsoluteScaleRecovery()
{
    std::cerr << "AbsoluteScaleRecovery is destructed.\n";  
};


void AbsoluteScaleRecovery::runASR(
    const FramePtrVec& frames_t0, 
    const FramePtrVec& frames_u,  
    const FramePtrVec& frames_t1)
{
    std::cerr << colorcode::text_yellow <<colorcode::cout_bold
    << "\n====================== Run Absolute Scale Recovery ...  ======================\n";

    // Optimization parameters
    int   MAX_ITER    = 7; // Maximum allowable iterations
    float THRES_HUBER = 0.5f; // huber threshold

    // The number of frames
    int K  = frames_t0.size() + frames_t1.size() - 1;
    int N  = frames_t0.size() + frames_t1.size() + frames_u.size();
    
    std::cerr << "In 'runASR()', N: " << N << ", K: "<< K << std::endl;
    std::cerr << " (The first frame of 'frame_t0' is fixed to prevent gauge freedom.)" << std::endl;

    // Generate frames-scales constraints 
    FramePtrVec         frames_t;
    std::vector<double> scales_t;
    for(int i = 1; i < frames_t0.size(); ++i) frames_t.push_back(frames_t0.at(i));
    for(int i = 0; i < frames_t1.size(); ++i) frames_t.push_back(frames_t1.at(i));
    for(const FramePtr& f : frames_t) scales_t.push_back(f->getScale());

    if( scales_t.size() != frames_t.size() )
        throw std::runtime_error("In 'runASR()', scales_t.size() != frames_t.size()");

    // Generate all frames and fix / non-fix index
    FramePtrVec frames_all;
    std::vector<int> idx_fix;
    std::vector<int> idx_opt;
    for(int i = 0; i < frames_t0.size(); ++i) frames_all.push_back(frames_t0.at(i));
    for(int i = 0; i <  frames_u.size(); ++i) frames_all.push_back(frames_u.at(i));
    for(int i = 0; i < frames_t1.size(); ++i) frames_all.push_back(frames_t1.at(i));
    
    idx_fix.push_back(0); // Set the first frame fixed.
    for(int i = 1; i < N; ++i) idx_opt.push_back(i); // The others are in optimization.

    std::cerr 
        << " N : " << N 
        << ", frames_all.size(): " << frames_all.size()
        << ", idx_fix.size(): " << idx_fix.size() 
        << ", idx_opt.size(): " << idx_opt.size() 
        << ", idx_fix.size() + idx_opt.size() : " << idx_fix.size() + idx_opt.size() << std::endl;

    if( N != frames_all.size() || N != (idx_fix.size() + idx_opt.size()) )
        throw std::runtime_error("In 'runASR()', N != frames_all.size() || N != (idx_fix.size() + idx_opt.size()).");


    // Make SQP constraint parameter (Sequential Quadratic Programming)
    std::shared_ptr<ScaleConstraints> scale_constraints;
    scale_constraints = std::make_shared<ScaleConstraints>();
    scale_constraints->setConstraints(frames_t, scales_t);

    // Make Sparse SQP parameters
    std::shared_ptr<SparseBAParameters> ba_params;
    ba_params = std::make_shared<SparseBAParameters>();
    ba_params->setPosesAndPoints(frames_all, idx_fix, idx_opt);

    timer::tic();
    sqp_solver_->reset();
    sqp_solver_->setCamera(cam_);
    sqp_solver_->setHuberThreshold(THRES_HUBER);
    sqp_solver_->setBAParametersAndConstraints(ba_params, scale_constraints);
    double dt_prepare = timer::toc(0);

    timer::tic();
    sqp_solver_->solveForFiniteIterations(MAX_ITER);
    double dt_solve = timer::toc(0);

    timer::tic();
    sqp_solver_->reset();
    double dt_reset = timer::toc(0);

    std::cout << "==== Show Translations: \n";
    for(int j = 0; j < ba_params->getNumOfOptimizeFrames(); ++j)
    {
        const FramePtr& f = ba_params->getOptFramePtr(j);

        std::cout << "[" << f->getID() << "] frame's trans: " << f->getPose().block<3,1>(0,3).transpose() << "\n";
    }

    

    // std::cout << "==== Show Points: \n";
    // for(int i = 0; i < ba_params->getNumOfOptimizeLandmarks(); ++i)
    // {
    //     const LandmarkPtr& lm = ba_params->getOptLandmarkPtr(i);
        
    //     if(lm->get3DPoint().norm() > 200)
    //         std::cout << " LARGE !!!! [" << lm->getID() << "] point: " << lm->get3DPoint().transpose() << "\n";
    //     else    
    //         std::cout << "[" << lm->getID() << "] point: " << lm->get3DPoint().transpose() << "\n";
    // }

    for(const auto& f : ba_params->getAllFrameset())
    {
        std::cout << f->getID() << "-th frame is " << 
        (f->isPoseOnlySuccess() ? "TRUE" : "FALSE") << std::endl;
    }
    
    // Time analysis
    std::cout << "==== SQP time to prepare: " << dt_prepare << " [ms]\n";
    std::cout << "==== SQP time to solve: "   << dt_solve   << " [ms]\n";
    std::cout << "==== SQP time to reset: "   << dt_reset   << " [ms]\n\n";
    std::cerr << "==== Absolute Scale Recovery is done! ====\n";
    std::cerr << "\n\n\n\n\n" << colorcode::cout_reset;

};