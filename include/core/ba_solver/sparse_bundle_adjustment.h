#ifndef _SPARSE_BUNDLE_ADJUSTMENT_H_
#define _SPARSE_BUNDLE_ADJUSTMENT_H_

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

typedef std::vector<Mat66>              BlockDiagMat66; 
typedef std::vector<Mat33>              BlockDiagMat33; 
typedef std::vector<std::vector<Mat63>> BlockFullMat63; 
typedef std::vector<std::vector<Mat36>> BlockFullMat36;
typedef std::vector<std::vector<Mat66>> BlockFullMat66;
typedef std::vector<Vec6>               BlockVec6;
typedef std::vector<Vec3>               BlockVec3;

/*
    <Problem we want to solve>
        H*delta_theta = J.transpose()*r;

    where
    - Hessian  
            H = [A_,  B_;
                 Bt_, C_];

    - Jacobian multiplied by residual vector 
            J.transpose()*r = [a;b];

    - Update parameters
            delta_theta = [x;y];
*/

struct LandmarkBA {
    Point             X;
    FramePtrVec       kfs_seen; // 해당 키프레임에서 어떤 좌표로 보였는지를 알아야 함.
    std::vector<int>  kfs_index; // 해당 키프레임이 window에서 몇번째인가 저장.
    PixelVec          pts_on_kfs; // 각 키프레임에서 추적된 pixel 좌표.
    
    LandmarkPtr       lm;

    LandmarkBA() {
        lm = nullptr;
        X  = Vec3::Zero();
    };
    LandmarkBA(const LandmarkBA& lmba) {
        lm              = lmba.lm;
        X               = lmba.X;
        kfs_seen        = lmba.kfs_seen;
        kfs_index       = lmba.kfs_index;
        pts_on_kfs      = lmba.pts_on_kfs;
    };
};

class SparseBAParameters {
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
    std::vector<LandmarkBA>    lmbavec_all_; // All landmarks to be optimized

    std::map<FramePtr,PoseSE3> posemap_all_; // pose map Tjw

    std::map<FramePtr,int>     indexmap_opt_; // optimization pose index map
    std::vector<FramePtr>      framemap_opt_; // j-th optimization frame ptr

// Get methods (numbers)
public:
    inline int getNumOfAllFrames()         const { return N_; };
    inline int getNumOfOptimizeFrames()    const { return N_opt_; };
    inline int getNumOfFixedFrames()       const { return N_fix_; };
    inline int getNumOfOptimizeLandmarks() const { return M_; };
    inline int getNumOfObservations()      const { return n_obs_; };

// Get methods (variables)
public:
    const PoseSE3& getPose(const FramePtr& frame) {
        if(posemap_all_.find(frame) == posemap_all_.end())
            throw std::runtime_error("posemap_all_.find(frame) == posemap_all_.end()");
        return posemap_all_[frame];
    };
    int getOptPoseIndex(const FramePtr& frame){
        if(indexmap_opt_.find(frame) == indexmap_opt_.end())
            throw std::runtime_error("indexmap_opt_.find(frame) == indexmap_opt_.end()");
        return indexmap_opt_[frame];
    };
    const FramePtr& getOptFramePtr(int j_opt){
        if( j_opt >= framemap_opt_.size())
            throw std::runtime_error("j_opt >= framemap_opt_.size()");
        return framemap_opt_[j_opt];
    };

    const LandmarkBA& getLandmarkBA(int i){
        if( i >= lmbavec_all_.size())
            throw std::runtime_error("i >= lmbavec_all_.size()");
        return lmbavec_all_[i];
    };

// Update and get methods (Pose and Point)
public:
    void updateOptPoint(int i, const Point& X_update){
        if(i >= M_)
            throw std::runtime_error("i >= M_");
        lmbavec_all_[i].X = X_update;
    };
    void updateOptPose(int j_opt, const PoseSE3& Tjw_update){
        if(j_opt >= N_opt_) 
            throw std::runtime_error("j_opt >= N_opt_");
        const FramePtr& kf_opt = framemap_opt_[j_opt];
        posemap_all_[kf_opt] = Tjw_update;
    };  

    const Point& getOptPoint(int i){
        if(i >= M_)
            throw std::runtime_error("i >= M_");
        return lmbavec_all_[i].X;
    };
    const PoseSE3& getOptPose(int j_opt){
        if(j_opt >= N_opt_) 
            throw std::runtime_error("j_opt >= N_opt_");
        const FramePtr& kf_opt = framemap_opt_[j_opt];
        return posemap_all_[kf_opt];
    };

// Find methods
public:
    bool isOptFrame(const FramePtr& f) {return (indexmap_opt_.find(f) != indexmap_opt_.end() ); };
    bool isFixFrame(const FramePtr& f) {return (indexmap_opt_.find(f) == indexmap_opt_.end() ); };


// Set methods
public:
    SparseBAParameters() 
    : N_(0), N_opt_(0), N_fix_(0), M_(0), n_obs_(0)
    { };

    ~SparseBAParameters(){};

    void setPosesAndPoints(
        const FramePtrVec&      frames, 
        const std::vector<int>& idx_fix, 
        const std::vector<int>& idx_optimize)
    {
        int THRES_MINIMUM_SEEN = 2;

        N_     = frames.size();

        N_fix_ = idx_fix.size();
        N_opt_ = idx_optimize.size();


        if( N_ != N_fix_ + N_opt_ ) 
            throw std::runtime_error(" N != N_fix + N_opt ");
        
        std::cout << "input N: " << N_ <<", N_fix + N_opt: " << N_fix_ << "+" << N_opt_ << std::endl;

        // 1) get all window keyframes 
        std::set<LandmarkPtr> lmset_window; // 안겹치는 랜드마크들
        std::set<FramePtr> frameset_window; // 윈도우 내부 키프레임들
        FramePtrVec kfvec_window; // 윈도우 내부의 키프레임들
        for(auto kf : frames) { // 모든 keyframe in window 순회 
            kfvec_window.push_back(kf); // window keyframes 저장.
            frameset_window.insert(kf);

            for(auto lm : kf->getRelatedLandmarkPtr()){ // 현재 keyframe에서 보인 모든 landmark 순회
                if( lm->isTriangulated() ) // age > THRES, triangulate() == true 경우 포함.
                    lmset_window.insert(lm);
            }
        }

        // 2) make LandmarkBAVec
        for(auto lm : lmset_window) { // keyframe window 내에서 보였던 모든 landmark를 순회.
            LandmarkBA lm_ba;
            lm_ba.lm = lm; // landmark pointer
            lm_ba.X  = lm->get3DPoint();  // 3D point represented in the global frame.
            lm_ba.kfs_seen.reserve(10);   
            lm_ba.kfs_index.reserve(10); 
            lm_ba.pts_on_kfs.reserve(10);

            // 현재 landmark가 보였던 keyframes을 저장한다.
            for(int j = 0; j < lm->getRelatedKeyframePtr().size(); ++j) {
                const FramePtr& kf = lm->getRelatedKeyframePtr()[j];
                const Pixel&    pt = lm->getObservationsOnKeyframes()[j];
                if(frameset_window.find(kf) != frameset_window.end())
                {   //window keyframe만으로 제한
                    lm_ba.kfs_seen.push_back(kf);
                    lm_ba.pts_on_kfs.push_back(pt);
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
        std::cout << "Recomputed N: " << N_ <<", N_fix + N_opt: " << N_fix_ << "+" << N_opt_ << std::endl;

        // 4) set poses for all frames
        for(auto kf : frameset_all_)
            posemap_all_.insert({kf, kf->getPoseInv()});
        
        // 5) set optimizable keyframes (posemap, indexmap, framemap)
        int cnt_idx = 0;
        idx_optimize;
        for(int jj = 0; jj < idx_optimize.size(); ++jj){
            int j = idx_optimize.at(jj);
            indexmap_opt_.insert({frames[j],cnt_idx});
            framemap_opt_.push_back(frames[j]);
            ++cnt_idx;
        }

        // 6) set optimization values 
        n_obs_ = 0; // the number of total observations (2*n_obs == len_residual)
        for(auto lm_ba : lmbavec_all_) 
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

private:


};

// A sparse solver for a feature-based Bundle adjustment problem.
class SparseBundleAdjustmentSolver{
private:
    std::shared_ptr<Camera> cam_;

private:
    // Storages to solve Schur Complement
    BlockDiagMat66 A_; // N_opt (6x6) block diagonal part for poses +
    BlockFullMat63 B_; // N_opt x M (6x3) block part (side) +
    BlockFullMat36 Bt_; // M x N_opt (3x6) block part (side, transposed) +
    BlockDiagMat33 C_; // M (3x3) block diagonal part for landmarks' 3D points +
    
    BlockVec6 a_; // N_opt x 1 (6x1) +
    BlockVec3 b_; // M x 1 (3x1) +

    BlockVec6 x_; // N_opt  (6x1)+
    BlockVec3 y_; // M  (3x1) +

    BlockVec6 params_poses_;  // N_opt (6x1) parameter vector for poses
    BlockVec3 params_points_; // M     (3x1) parameter vector for points

    BlockDiagMat33 Cinv_;    // M (3x3) block diagonal part for landmarks' 3D points (inverse) +
    BlockFullMat63 BCinv_;   // N_opt X M  (6x3) +
    BlockFullMat36 CinvBt_;  // M x N_opt (3x6) +
    BlockFullMat66 BCinvBt_; // N_opt x N_opt (6x6) +
    BlockVec6      BCinv_b_; // N_opt (6x1) +
    BlockVec3      Bt_x_;    // M     (3x1) +
    BlockVec3      Cinv_b_;  // M     (3x1) +

    BlockFullMat66 Am_BCinvBt_; // N_opt x N_opt (6x6) +
    BlockVec6      am_BCinv_b_;      // N_opt (6x1) +
    BlockVec3      CinvBt_x_;        // M (3x1) +

    // Input variable
    std::shared_ptr<SparseBAParameters> ba_params_;

    // Input variables  
    // std::vector<LandmarkBA>    lms_ba_; // landmarks to be optimized
    // std::map<FramePtr,PoseSE3> Tjw_map_; // map containing poses (for all keyframes including fixed ones)
    // std::map<FramePtr,int>     kfmap_optimize_; // map containing optimizable keyframes and their indexes
    // std::vector<FramePtr>      kfvec_optimize_;

private:
    // Problem sizes
    int N_;     // # of total poses including fixed poses
    int N_opt_; // # of poses to be optimized
    int M_;     // # of landmarks to be optimized
    int n_obs_;

private:
    // Optimization parameters
    float THRES_HUBER_;
    float THRES_EPS_;

public:
    // Constructor for BundleAdjustmentSolver ( se3 (6-DoF), 3D points (3-DoF) )
    // Sparse solver with 6x6 block diagonals, 3x3 block diagonals, 6x3 and 3x6 blocks.
    SparseBundleAdjustmentSolver();

    // Set connectivities, variables...
    void setBAParameters(const std::shared_ptr<SparseBAParameters>& ba_params);

    // Set Huber threshold
    void setHuberThreshold(float thres_huber);

    // Set camera.
    void setCamera(const std::shared_ptr<Camera>& cam);

    // Solve the BA for fixed number of iterations
    bool solveForFiniteIterations(int MAX_ITER);

    // Reset local BA solver.
    void reset();

private:
    // Set problem sizes and resize the storages.
    void setProblemSize(int N, int N_opt, int M, int n_obs);

// Related to parameter vector
private:

    void setParameterVectorFromPosesPoints();
    void getPosesPointsFromParameterVector();
    void zeroizeStorageMatrices();    
};
#endif