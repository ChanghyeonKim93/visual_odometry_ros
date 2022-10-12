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


void AbsoluteScaleRecovery::runASR()
{
    //
};