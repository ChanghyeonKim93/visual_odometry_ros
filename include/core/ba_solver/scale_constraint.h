#ifndef _SCALE_CONSTRAINT_H_
#define _SCALE_CONSTRAINT_H_

#include <iostream>
#include <vector>
#include <map>
#include <set>

#include <Eigen/Dense>

#include "core/camera.h"
#include "core/landmark.h"
#include "core/frame.h"
#include "core/type_defines.h"

#include "util/geometry_library.h"
#include "util/timer.h"

typedef double                           _BA_numeric; 
typedef int                              _BA_Index;
typedef int                              _BA_Int;
typedef _BA_numeric                      _BA_Scale;

typedef Eigen::Matrix<_BA_numeric,-1,-1> _BA_MatX;

typedef Eigen::Matrix<_BA_numeric,2,2>   _BA_Mat22;

typedef Eigen::Matrix<_BA_numeric,2,3>   _BA_Mat23;
typedef Eigen::Matrix<_BA_numeric,3,2>   _BA_Mat32;

typedef Eigen::Matrix<_BA_numeric,3,3>   _BA_Mat33;

typedef Eigen::Matrix<_BA_numeric,2,1>   _BA_Vec2;
typedef Eigen::Matrix<_BA_numeric,3,1>   _BA_Vec3;

typedef Eigen::Matrix<_BA_numeric,2,1>   _BA_Pixel;
typedef Eigen::Matrix<_BA_numeric,3,1>   _BA_Point;
typedef Eigen::Matrix<_BA_numeric,3,3>   _BA_Rot3;
typedef Eigen::Matrix<_BA_numeric,3,1>   _BA_Pos3;
typedef Eigen::Matrix<_BA_numeric,4,4>   _BA_PoseSE3;
typedef Eigen::Matrix<_BA_numeric,6,1>   _BA_PoseSE3Tangent;

typedef std::vector<_BA_Index>           _BA_IndexVec;
typedef std::vector<_BA_Scale>           _BA_ScaleVec;
typedef std::vector<_BA_Pixel>           _BA_PixelVec;
typedef std::vector<_BA_Point>           _BA_PointVec;

typedef std::vector<_BA_Mat33>              BlockDiagMat33; 
typedef std::vector<std::vector<_BA_Mat33>> BlockFullMat33; 
typedef std::vector<_BA_Vec3>               BlockVec3;


// 각 프레임이 스케일을 가지고 있는 형태로 가야하나 ? 
// 스케일을 따로 넣어주는 형태로 가야하나 ? 

/// @brief parameter class for Translation Scale Constraints 
class ScaleConstraints
{
private: // scaling factor for numerical stability
    _BA_Int N_t_; // total number of constraint frames
    _BA_numeric scaler_;
    _BA_numeric inv_scaler_;

    std::set<FramePtr> frameset_; // constraint를 가지는 프레임들
    std::map<FramePtr, _BA_Scale> scalemap_; // scalemap all, frame을 넣으면 scale이 나온다.
    
// Get methods (numbers)
public:
    inline _BA_Int getNumOfConstraint() const { return N_t_; };
    
// Get methods (variables)
public:
    const _BA_Scale& getScale(const FramePtr& frame) 
    {
        if(scalemap_.find(frame) == scalemap_.end())
            throw std::runtime_error("scalemap_.find(frame) == scalemap_.end()");
        return scalemap_.at(frame);
    };

    const std::set<FramePtr>& getAllConstraintFrameset() { return frameset_; };
    const std::map<FramePtr,_BA_Scale>& getScalemap() { return scalemap_; };

// Find methods
public:
    bool isConstraintFrame(const FramePtr& f) {return (frameset_.find(f) != frameset_.end() ); };
    
// Set methods
public:
    ScaleConstraints() 
    : N_t_(0), scaler_(10.0), inv_scaler_(1.0/scaler_)
    { 
        std::cerr << "ScaleConstraints is constructed.\n";
    };

    ~ScaleConstraints() {
        std::cerr << "ScaleConstraints is deleted.\n";
    };

    void setScaleConstraints(
        const FramePtrVec&  frames, 
        const _BA_ScaleVec& scales)
    {
        N_t_ = frames.size();

        for(int j = 0; j < N_t_; ++j)
        {
            frameset_.insert(frames[j]);
            scalemap_.insert({frames[j], scales[j]*inv_scaler_});
        }

        printf("| Scale Constraints Statistics:\n");
        printf("|  -   # of constraint images: %d images \n", N_t_);
        printf("|  -           numeric scaler: %0.3f times scaled\n", inv_scaler_);
    };
};


#endif