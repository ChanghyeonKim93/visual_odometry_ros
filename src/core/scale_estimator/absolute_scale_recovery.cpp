#include "core/scale_estimator/absolute_scale_recovery.h"

AbsoluteScaleRecovery::AbsoluteScaleRecovery(const std::shared_ptr<Camera>& cam)
: cam_(cam)
{

    std::cerr << "AbsoluteScaleRecovery is constructed.\n";
};


AbsoluteScaleRecovery::~AbsoluteScaleRecovery()
{
    std::cerr << "AbsoluteScaleRecovery is destructed.\n";  
};


void AbsoluteScaleRecovery::runASR(
    const FramePtrVec& frames_t0, 
    const FramePtrVec& frames_u, 
    const FramePtrVec& frames_t1
)
{
    std::cerr << "== Run Absolute Scale Recovery ...\n";

    // Optimization paraameters
    int   MAX_ITER          = 20;
    float THRES_HUBER       = 0.5f;


    // Make SQP param (Sequential Quadratic Programming)
    int Nt = frames_t0.size() + frames_t1.size() - 1;
    int N  = frames_t0.size() + frames_t1.size() + frames_u.size();

    std::cerr << "In runASR, N: " << N << ", Nt: "<< Nt << std::endl;
    std::cerr << " (The first frame of 'frame_t0' is fixed to prevent gauge freedom.)" << std::endl;

    FramePtrVec frames_t;
    std::vector<double> scales_t;
    for(int i = 1; i < frames_t0.size(); ++i) frames_t.push_back(frames_t0.at(i));
    for(const FramePtr& f : frames_t1) frames_t.push_back(f);

    for(const FramePtr& f : frames_t) scales_t.push_back(f->getScale());

    if( scales_t.size() != frames_t.size() )
        throw std::runtime_error("scales_t.size() != frames_t.size()");

    std::shared_ptr<ScaleConstraints> scale_constraints;
    scale_constraints = std::make_shared<ScaleConstraints>();
    scale_constraints->setScaleConstraints(frames_t,scales_t);

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