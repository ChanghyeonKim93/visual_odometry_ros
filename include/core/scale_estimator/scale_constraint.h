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
#include "core/ba_solver/ba_types.h"

#include "util/geometry_library.h"
#include "util/timer.h"

typedef int                              _BA_Int;
typedef _BA_numeric                      _BA_Scale;

typedef std::vector<_BA_Scale>           _BA_ScaleVec;

// {k}-th constraint related to {j}-th and {j-1} th frames

// getConstIndex:    Fj 또는 Fj-1을 쿼리 --> k 출력
// getRelatedFrames: k 번째 constraint 를 쿼리 --> {Fj, Fj-1} pair 출력 (major and prev.)
// 


class RelatedFramePair
{
private:
    FramePtr f_major_;
    FramePtr f_minor_;

public:
    RelatedFramePair() 
    : f_major_(nullptr), f_minor_(nullptr) { };

    RelatedFramePair(const FramePtr& f_major, const FramePtr& f_minor)
    : f_major_(f_major), f_minor_(f_minor) { };
    
public:
    const FramePtr& getMajor() const {return f_major_; };
    const FramePtr& getMinor() const {return f_minor_; };
};

/*
ConstValue       : Constraint value (scale)
ConstIndex       : Constraint index (int)
RelatedFramePair : Constrained two frames (FramePtr)


std::map<FramePtr, _BA_Scale> frame_to_scale_map_;
std::map<FramePtr, _BA_Index> frame_to_index_map_;
std::vector<RelatedFramePair> index_to_framepair_map_;
std::map<FramePtr, FramePtr> fmajor_to_fminor_map_;

    getConstValueByMajorFrame(f) : Fj -> sk (throw std::runtime_error)
    getConstIndexByMajor(f) : Fj -> k (throw std::runtime_error)

    getRelatedFramePairByConstIndex : k -> {Fj, Fj-1} (Constraint Index로 FramePair를 불러온다.)
    getMajorByMinor(f) : F(j-1) -> Fj (Major frame으로 minor Frame을 불러온다.)
    getMinorByMajor(f) : Fj -> F(j-1) (Minor frame으로 Major frame을 불러온다.)
    

    
*/

/// @brief parameter class for Translation Scale Constraints 
class ScaleConstraints
{
private: // scaling factor for numerical stability
    _BA_Int Nt_; // total number of constraint frames
    _BA_numeric scaler_;
    _BA_numeric inv_scaler_;

private: // all frames constrained
    std::set<FramePtr> frameset_; // constraint를 가지는 프레임들


private: 
    std::map<FramePtr, _BA_Scale> frame_to_scale_map_;
    std::map<FramePtr, _BA_Index> frame_to_index_map_;
    std::vector<RelatedFramePair> index_to_framepair_map_;
    std::map<FramePtr, FramePtr> fmajor_to_fminor_map_;
    std::map<FramePtr, FramePtr> fminor_to_fmajor_map_;


    // std::map<FramePtr, _BA_Scale> scalemap_; // scalemap all, frame을 넣으면 scale이 나온다.    
    // std::map<FramePtr, _BA_Index> indexmap_const_; // constraint index map (몇 번째 constraint인가)



// Constructor and destructor
public:
    ScaleConstraints() 
    : Nt_(0), scaler_(10.0), inv_scaler_(1.0/scaler_)
    { 
        std::cerr << "          ScaleConstraints is constructed.\n";
    };

    ~ScaleConstraints() {
        std::cerr << "          ScaleConstraints is deleted.\n";
    };    

// Set methods
public:
    void setConstraints(
        const FramePtrVec& frames, const _BA_ScaleVec& scales)
    {

        if(frames.size() != scales.size())
            throw std::runtime_error("In 'setConstraints()', frames.size() != scales.size()");

        if(frames.empty())
            throw std::runtime_error("In 'setConstraints()', frames.size() == 0 (no constraint is given.)");

        // The number of all input frames 
        this->Nt_ = frames.size();

        for(int j = 0; j < this->Nt_; ++j)
        {
            const FramePtr& f_major = frames.at(j);
            const FramePtr& f_minor = f_major->getPreviousTurningFrame();

            frameset_.insert(f_major);
            frame_to_scale_map_.insert({f_major, scales[j]*inv_scaler_});
            frame_to_index_map_.insert({f_major, j});
            index_to_framepair_map_.emplace_back(f_major, f_minor);
            fmajor_to_fminor_map_.insert({f_major, f_minor});
            fminor_to_fmajor_map_.insert({f_minor, f_major});
        }

        // Check ! 
        std::cout << "          frameset_.size():"               << frameset_.size() << std::endl;
        std::cout << "          frame_to_scale_map_.size():"     << frame_to_scale_map_.size() << std::endl;
        std::cout << "          frame_to_index_map_.size():"     << frame_to_index_map_.size() << std::endl;
        std::cout << "          index_to_framepair_map_.size():" << index_to_framepair_map_.size() << std::endl;
        std::cout << "          fmajor_to_fminor_map_.size():"   << fmajor_to_fminor_map_.size() << std::endl;
        std::cout << "          fminor_to_fmajor_map_.size():"   << fminor_to_fmajor_map_.size() << std::endl;

        printf("| Scale Constraints Statistics:\n");
        printf("|  -   # of constraint images: %d images \n", Nt_);
        printf("|  -           numeric scaler: %0.3f times scaled\n", inv_scaler_);
    };

// Get methods (numbers)
public:
    inline _BA_Int getNumOfConstraint() const { return Nt_; };
    
// Get methods (variables)
public:
    const std::set<FramePtr>& getAllConstraintFrameset() { return frameset_; };
    // const std::map<FramePtr,_BA_Scale>& getScalemap() { return frame_to_scale_map_; };
    // const _BA_Scale& getScale(const FramePtr& frame) 
    // {
    //     if(scalemap_.find(frame) == scalemap_.end())
    //         throw std::runtime_error("scalemap_.find(frame) == scalemap_.end()");
    //     return scalemap_.at(frame);
    // };
    // _BA_Index getConstIndex(const FramePtr& frame){
    //     if(sc.find(frame) == indexmap_opt_.end())
    //         throw std::runtime_error("indexmap_opt_.find(frame) == indexmap_opt_.end()");
    //     return indexmap_opt_.at(frame);
    // };

    inline _BA_Scale getConstValueByMajorFrame(const FramePtr& f) const 
    { 
        if(frame_to_scale_map_.find(f) == frame_to_scale_map_.end())
            throw std::runtime_error("frame_to_scale_map_.find(f) == frame_to_scale_map_.end()");
        return frame_to_scale_map_.at(f);
    };

    inline _BA_Index getConstIndexByMajorFrame(const FramePtr& f) const
    {
        if(frame_to_index_map_.find(f) == frame_to_index_map_.end())
            throw std::runtime_error("frame_to_index_map_.find(f) == frame_to_index_map_.end()");
        return frame_to_index_map_.at(f);
    };

    inline const RelatedFramePair& getRelatedFramePairByConstIndex(_BA_Index k) const 
    {
        if(k >= Nt_ || k < 0)
            throw std::runtime_error("In 'getRelatedFramePairByConstIndex()', k >= Nt_ || k < 0 ");
        return index_to_framepair_map_.at(k);
    };

    inline const FramePtr& getMajorByMinor(const FramePtr& f_major) const 
    {
        if(frame_to_index_map_.find(f_major) == frame_to_index_map_.end())
            throw std::runtime_error("In 'getMajorByMinor()', frame_to_index_map_.find(f_major) == frame_to_index_map_.end()");
        return fmajor_to_fminor_map_.at(f_major);
    };

    inline const FramePtr& getMinorByMajor(const FramePtr& f_minor) const 
    {
        if(fminor_to_fmajor_map_.find(f_minor) == fminor_to_fmajor_map_.end())
            throw std::runtime_error("In 'getMinorByMajor()', fminor_to_fmajor_map_.find(f_minor) == fminor_to_fmajor_map_.end()");
        return fminor_to_fmajor_map_.at(f_minor);
    };
    
// Find methods
public:
    bool isConstrainedFrame(const FramePtr& f) {return (frameset_.find(f) != frameset_.end() ); };

};


#endif