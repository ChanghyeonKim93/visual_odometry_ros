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
    std::cerr << "== Run Absolute Scale Recovery ...\n";

    // Optimization parameters
    int   MAX_ITER    = 20; // Maximum allowable iterations
    float THRES_HUBER = 0.5f; // huber threshold

    // The number of frames
    int Nt = frames_t0.size() + frames_t1.size() - 1;
    int N  = frames_t0.size() + frames_t1.size() + frames_u.size();
    
    std::cerr << "In runASR, N: " << N << ", Nt: "<< Nt << std::endl;
    std::cerr << " (The first frame of 'frame_t0' is fixed to prevent gauge freedom.)" << std::endl;

    // Generate frames-scales constraints 
    FramePtrVec         frames_t;
    std::vector<double> scales_t;
    for(int i = 1; i < frames_t0.size(); ++i) frames_t.push_back(frames_t0.at(i));
    for(int i = 0; i < frames_t1.size(); ++i) frames_t.push_back(frames_t1.at(i));
    for(const FramePtr& f : frames_t) scales_t.push_back(f->getScale());

    if( scales_t.size() != frames_t.size() )
        throw std::runtime_error("scales_t.size() != frames_t.size()");

    // Make SQP constraint parameter (Sequential Quadratic Programming)
    std::shared_ptr<ScaleConstraints> scale_constraints;
    scale_constraints = std::make_shared<ScaleConstraints>();
    scale_constraints->setScaleConstraints(frames_t, scales_t);

    // Make Sparse SQP parameters
    // std::shared_ptr<SparseBAParameters> ba_params;
    // ba_params = std::make_shared<SparseBAParameters>();
    // ba_params->setPosesAndPoints(frames, idx_fix, idx_opt);

    timer::tic();
    double dt_prepare = timer::toc(0);

    timer::tic();
    double dt_solve = timer::toc(0);

    timer::tic();
    double dt_reset = timer::toc(0);

    // Time analysis
    std::cout << "==== SQP time to prepare: " << dt_prepare << " [ms]\n";
    std::cout << "==== SQP time to solve: "   << dt_solve   << " [ms]\n";
    std::cout << "==== SQP time to reset: "   << dt_reset   << " [ms]\n\n";
    std::cerr << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";

    std::cerr << "== Absolute Scale Recovery is done!\n";
};