#ifndef _SPARSE_BA_PARAMETERS_H_
#define _SPARSE_BA_PARAMETERS_H_
#include <iostream>
#include <vector>
#include <map>
#include <set>

#include <Eigen/Dense>

#include "core/landmark.h"
#include "core/frame.h"
#include "core/type_defines.h"

#include "util/geometry_library.h"

#include "core/ba_solver/ba_types.h"
#include "core/ba_solver/landmark_ba.h"

/// @brief parameter class for Sparse Bundle Adjustment
class SparseBAParameters 
{
private: // Reference Pose and scaling factor for numerical stability
    _BA_PoseSE3 Twj_ref_;
    _BA_PoseSE3 Tjw_ref_;
    _BA_numeric pose_scale_;
    _BA_numeric inv_pose_scale_;

private: // all frames and landmarks used for BA.
    std::set<FramePtr>       frameset_all_;
    std::set<LandmarkPtr>    landmarkset_all_;

private: 
    int N_; // total number of frames
    int N_opt_; // # of optimizable frames
    int N_fix_; // # of fixed frames (to prevent gauge freedom)

    int M_; // total number of landmarks (all landmarks is to be optimized if they satisty the conditions.) 

    int n_obs_; // the number of observations

private:
    std::vector<LandmarkBA> lmbavec_all_; // All landmarks to be optimized

    std::map<FramePtr,_BA_PoseSE3> posemap_all_; // pose map Tjw

    std::map<FramePtr,_BA_Index> indexmap_opt_; // optimization pose index map

    FramePtrVec  framemap_opt_; // j-th optimization frame ptr

// Get methods (numbers)
public:
    inline int getNumOfAllFrames()         const { return N_; };
    inline int getNumOfOptimizeFrames()    const { return N_opt_; };
    inline int getNumOfFixedFrames()       const { return N_fix_; };
    inline int getNumOfOptimizeLandmarks() const { return M_; };
    inline int getNumOfObservations()      const { return n_obs_; };

// Get methods (variables)
public:
    const _BA_PoseSE3& getPose(const FramePtr& frame) {
        if(posemap_all_.find(frame) == posemap_all_.end())
            throw std::runtime_error("posemap_all_.find(frame) == posemap_all_.end()");
        return posemap_all_.at(frame);
    };
    _BA_Index getOptPoseIndex(const FramePtr& frame){
        if(indexmap_opt_.find(frame) == indexmap_opt_.end())
            throw std::runtime_error("indexmap_opt_.find(frame) == indexmap_opt_.end()");
        return indexmap_opt_.at(frame);
    };
    const FramePtr& getOptFramePtr(_BA_Index j_opt){
        if( j_opt >= framemap_opt_.size())
            throw std::runtime_error("j_opt >= framemap_opt_.size()");
        return framemap_opt_.at(j_opt);
    };

    const LandmarkBA& getLandmarkBA(_BA_Index i){
        if( i >= lmbavec_all_.size())
            throw std::runtime_error("i >= lmbavec_all_.size()");
        return lmbavec_all_.at(i);
    };

    // Reference version
    LandmarkBA& getLandmarkBARef(_BA_Index i){
        if( i >= lmbavec_all_.size())
            throw std::runtime_error("i >= lmbavec_all_.size()");
        return lmbavec_all_.at(i);
    };

    const std::set<FramePtr>& getAllFrameset(){
        return frameset_all_;
    };
    const std::set<LandmarkPtr>& getAllLandmarkset(){
        return landmarkset_all_;
    };

    const std::map<FramePtr,_BA_PoseSE3>& getPosemap(){
        return posemap_all_;
    };

// Update and get methods (Pose and Point)
public:
    void updateOptPoint(_BA_Index i, const _BA_Point& X_update){
        if(i >= M_ || i < 0)
            throw std::runtime_error("i >= M_ || i < 0");
        lmbavec_all_[i].X(0) = X_update(0);
        lmbavec_all_[i].X(1) = X_update(1);
        lmbavec_all_[i].X(2) = X_update(2);
    };
    
    void updateOptPose(_BA_Index j_opt, const _BA_PoseSE3& Tjw_update){
        if(j_opt >= N_opt_ || j_opt < 0) 
            throw std::runtime_error("j_opt >= N_opt_  || j_opt < 0");

        const FramePtr& kf_opt = framemap_opt_.at(j_opt);
        posemap_all_.at(kf_opt) = Tjw_update;
        
        if(std::isnan(Tjw_update.norm()))
            throw std::runtime_error("Tjw update nan!");
    };

    const _BA_Point& getOptPoint(_BA_Index i){
        if(i >= M_ || i < 0)
            throw std::runtime_error("i >= M_ || i < 0");
        return lmbavec_all_[i].X;
    };

    const _BA_PoseSE3& getOptPose(_BA_Index j_opt){
        if(j_opt >= N_opt_ || j_opt < 0) 
            throw std::runtime_error("j_opt >= N_opt_ || j_opt < 0");
        const FramePtr& kf_opt = framemap_opt_.at(j_opt);
        return posemap_all_.at(kf_opt);
    };

// Find methods
public:
    bool isOptFrame(const FramePtr& f) {return (indexmap_opt_.find(f) != indexmap_opt_.end() ); };
    bool isFixFrame(const FramePtr& f) {return (indexmap_opt_.find(f) == indexmap_opt_.end() ); };

public:
    _BA_Point warpToRef(const _BA_Point& X){
        _BA_Point Xw = Tjw_ref_.block<3,3>(0,0)*X + Tjw_ref_.block<3,1>(0,3);
        return Xw;
    };
    _BA_Point warpToWorld(const _BA_Point& X){
        _BA_Point Xw = Twj_ref_.block<3,3>(0,0)*X + Twj_ref_.block<3,1>(0,3);
        return Xw;
    };

    _BA_PoseSE3 changeInvPoseWorldToRef(const _BA_PoseSE3& Tjw){
        _BA_PoseSE3 Tjref = Tjw*Twj_ref_;
        return Tjref;
    };
    _BA_PoseSE3 changeInvPoseRefToWorld(const _BA_PoseSE3& Tjref){
        _BA_PoseSE3 Tjw = Tjref*Tjw_ref_;
        return Tjw;
    };

    _BA_Point scalingPoint(const _BA_Point& X){
        _BA_Point X_scaled = X*inv_pose_scale_;
        return X_scaled;
    };
    _BA_Point recoverOriginalScalePoint(const _BA_Point& X){
        _BA_Point X_recovered = X*pose_scale_;
        return X_recovered;  
    };
    _BA_PoseSE3 scalingPose(const _BA_PoseSE3& Tjw){
        _BA_PoseSE3 Tjw_scaled = Tjw;
        Tjw_scaled(0,3) *= inv_pose_scale_;
        Tjw_scaled(1,3) *= inv_pose_scale_;
        Tjw_scaled(2,3) *= inv_pose_scale_;
        return Tjw_scaled;
    };
    _BA_PoseSE3 recoverOriginalScalePose(const _BA_PoseSE3& Tjw_scaled){
        _BA_PoseSE3 Tjw_org = Tjw_scaled;
        Tjw_org(0,3) *= pose_scale_;
        Tjw_org(1,3) *= pose_scale_;
        Tjw_org(2,3) *= pose_scale_;
        return Tjw_org;
    };

// Set methods
public:
    SparseBAParameters() 
    : N_(0), N_opt_(0), N_fix_(0), M_(0), n_obs_(0),
    pose_scale_(10.0), inv_pose_scale_(1.0/pose_scale_)
    { 

    };

    ~SparseBAParameters()
    {
        std::cout << "Sparse BA Parameters is deleted.\n";
    };

    void setPosesAndPoints(
        const FramePtrVec&  frames, 
        const _BA_IndexVec& idx_fix, 
        const _BA_IndexVec& idx_optimize)
    {
        // Threshold
        int THRES_MINIMUM_SEEN = 2;

        N_     = frames.size();

        N_fix_ = idx_fix.size();
        N_opt_ = idx_optimize.size();

        if( N_ != N_fix_ + N_opt_ ) 
            throw std::runtime_error(" N != N_fix + N_opt ");
        

        // 1) get all window keyframes 
        std::set<LandmarkPtr> lmset_window; // 안겹치는 랜드마크들
        std::set<FramePtr> frameset_window; // 윈도우 내부 키프레임들
        FramePtrVec kfvec_window; // 윈도우 내부의 키프레임들
        for(const auto& kf : frames) { // 모든 keyframe in window 순회 
            kfvec_window.push_back(kf); // window keyframes 저장.
            frameset_window.insert(kf);

            for(const auto& lm : kf->getRelatedLandmarkPtr()){ // 현재 keyframe에서 보인 모든 landmark 순회
                if( lm->isTriangulated() ) // age > THRES, triangulate() == true 경우 포함.
                    lmset_window.insert(lm);
            }
        }
        // 1-1) get reference pose.
        const PoseSE3& Twj_ref_float = kfvec_window.front()->getPose();
        Twj_ref_ << Twj_ref_float(0,0),Twj_ref_float(0,1),Twj_ref_float(0,2),Twj_ref_float(0,3),
                    Twj_ref_float(1,0),Twj_ref_float(1,1),Twj_ref_float(1,2),Twj_ref_float(1,3),
                    Twj_ref_float(2,0),Twj_ref_float(2,1),Twj_ref_float(2,2),Twj_ref_float(2,3),
                    Twj_ref_float(3,0),Twj_ref_float(3,1),Twj_ref_float(3,2),Twj_ref_float(3,3);
        
        Tjw_ref_ = geometry::inverseSE3(Twj_ref_);

        // 2) make LandmarkBAVec
        for(const auto& lm : lmset_window) { // keyframe window 내에서 보였던 모든 landmark를 순회.
            LandmarkBA lm_ba;
            lm_ba.lm = lm; // landmark pointer

            // warp to Reference frame.
            const Point& X_float = lm->get3DPoint(); 
            lm_ba.X << X_float(0), X_float(1), X_float(2);  // 3D point represented in the global frame.
            
            lm_ba.X = this->warpToRef(lm_ba.X);
            lm_ba.X = this->scalingPoint(lm_ba.X);
            
            lm_ba.kfs_seen.reserve(100);
            lm_ba.pts_on_kfs.reserve(100);

            // 현재 landmark가 보였던 keyframes을 저장한다.
            for(int j = 0; j < lm->getRelatedKeyframePtr().size(); ++j) {
                const FramePtr& kf = lm->getRelatedKeyframePtr()[j];
                const Pixel&    pt = lm->getObservationsOnKeyframes()[j];
                if(frameset_window.find(kf) != frameset_window.end())
                {   //window keyframe만으로 제한
                    lm_ba.kfs_seen.push_back(kf);
                    lm_ba.pts_on_kfs.emplace_back(pt.x, pt.y);
                }
            }

            // 충분히 많은 keyframes in window에서 보인 landmark만 최적화에 포함.
            if(lm_ba.kfs_seen.size() >= THRES_MINIMUM_SEEN) {
                lmbavec_all_.push_back(lm_ba); 
                landmarkset_all_.insert(lm);
                for(int j = 0; j < lm_ba.kfs_seen.size(); ++j) // all related keyframes.
                    frameset_all_.insert(lm_ba.kfs_seen[j]);
            }
        }

        // 3) re-initialize N, N_fix, M
        M_ = lmbavec_all_.size();
        N_ = frameset_all_.size(); 
        N_fix_ = N_ - N_opt_;
        // std::cout << "Recomputed N: " << N_ <<", N_fix + N_opt: " << N_fix_ << "+" << N_opt_ << std::endl;

        // 4) set poses for all frames
        for(const auto& kf : frameset_all_){
            _BA_PoseSE3 Tjw_tmp;
            const PoseSE3& Tjw_float = kf->getPoseInv();
            Tjw_tmp << Tjw_float(0,0), Tjw_float(0,1), Tjw_float(0,2), Tjw_float(0,3),
                       Tjw_float(1,0), Tjw_float(1,1), Tjw_float(1,2), Tjw_float(1,3),
                       Tjw_float(2,0), Tjw_float(2,1), Tjw_float(2,2), Tjw_float(2,3),
                       Tjw_float(3,0), Tjw_float(3,1), Tjw_float(3,2), Tjw_float(3,3);
            
            Tjw_tmp = this->changeInvPoseWorldToRef(Tjw_tmp);
            Tjw_tmp = this->scalingPose(Tjw_tmp);
            
            posemap_all_.insert(std::pair<FramePtr,_BA_PoseSE3>(kf, Tjw_tmp));
        }
        
        // 5) set optimizable keyframes (posemap, indexmap, framemap)
        _BA_Index cnt_idx = 0;
        idx_optimize;
        for(int jj = 0; jj < idx_optimize.size(); ++jj){
            int j = idx_optimize.at(jj);
            indexmap_opt_.insert({frames[j],cnt_idx});
            framemap_opt_.push_back(frames[j]);
            ++cnt_idx;
        }

        // 6) set optimization values 
        n_obs_ = 0; // the number of total observations (2*n_obs == len_residual)
        for(const auto& lm_ba : lmbavec_all_) 
            n_obs_ += lm_ba.kfs_seen.size(); // residual 크기.

        
        int len_residual  = 2*n_obs_;
        int len_parameter = 6*N_opt_ + 3*M_;


        printf("| Bundle Adjustment Statistics:\n");
        printf("|  -        # of total images: %d images \n", N_);
        printf("|  -           -  opt. images: %d images \n", N_opt_);
        printf("|  -           -  fix  images: %d images \n", N_fix_);
        printf("|  -        # of opti. points: %d landmarks \n", M_);
        printf("|  -        # of observations: %d \n", n_obs_);
        printf("|  -            Jacobian size: %d rows x %d cols\n", len_residual, len_parameter);
        printf("|  -            Residual size: %d rows\n\n", len_residual);
    };
};

#endif