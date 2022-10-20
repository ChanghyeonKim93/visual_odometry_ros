#include "core/landmark_tracking.h"

LandmarkTracking::LandmarkTracking()
{
    pts0.reserve(1000);
    pts1.reserve(1000);
    lms.reserve(1000);
    scale_change.reserve(1000);
    n_pts = 0;
};