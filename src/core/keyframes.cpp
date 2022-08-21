#include "core/keyframes.h"

Keyframes::Keyframes()
: THRES_OVERLAP_FEATURE_RATIO_(0.7f),
THRES_ROTATION_(2.0f*D2R),
THRES_TRANSLATION_(2.0f),
n_max_keyframes_(7)
{ 

};

void Keyframes::setMaxKeyframes(int max_kf){
    n_max_keyframes_ = max_kf;
};

void Keyframes::addNewKeyframe(const FramePtr& frame){
    if(kfs_list_.size() == n_max_keyframes_) {
        all_keyframes_.push_back(kfs_list_.front()); // keyframe window에서 제외된 키프레임들을 저장.
        kfs_list_.front()->outOfKeyframeWindow(); // keyframe window에서 제거됨을 표시.
        kfs_list_.pop_front(); // keyframe window에서 제거.
    }
    frame->makeThisKeyframe(); // 새 keyframe이 됨을 표시. (추가시에는 당연히 keyframe window로 들어옴)
    kfs_list_.push_back(frame); // 새 keyframe을 추가.

    // 새로 추가된 keyframe과 관련된 landmark 정보를 업데이트.
    const LandmarkPtrVec& lms = frame->getRelatedLandmarkPtr();
    for(int i = 0; i < lms.size(); ++i){
        const LandmarkPtr& lm = lms[i];
        lm->addObservationAndRelatedKeyframe(lm->getObservations().back(), frame);
    }
};

bool Keyframes::checkUpdateRule(const FramePtr& frame_curr){
    bool flag_addkeyframe = false;
    if(kfs_list_.empty()) flag_addkeyframe = true;

    if(!flag_addkeyframe){
        
        std::vector<float> tracking_ratios;
        for(std::list<FramePtr>::iterator it = kfs_list_.begin(); it != kfs_list_.end(); ++it){
            const FramePtr& kf = *it;
            // calculate tracking ratio
            int cnt_tracked = 0;
            int cnt_total_lms = 0;
            for(auto lm : kf->getRelatedLandmarkPtr()){
                if(lm->getRelatedFramePtr().back() == frame_curr) ++cnt_tracked;
                ++cnt_total_lms;
            }
            float tracking_ratio = (float)cnt_tracked/(float)cnt_total_lms;
            tracking_ratios.push_back(tracking_ratio);
        }
        std::cout << "       Tracking ratios : ";
        for(auto r : tracking_ratios){
            std::cout << (int)(r*100.f) << "% ";
        }
        std::cout << "\n";
        

        if(tracking_ratios.back() <= THRES_OVERLAP_FEATURE_RATIO_) flag_addkeyframe = true;
        
        // Check rotation & translation
        const PoseSE3& Twk_last = kfs_list_.back()->getPose();
        const PoseSE3& Twc = frame_curr->getPose();
        PoseSE3 dT = Twk_last.inverse()*Twc;
        if(!flag_addkeyframe){
            // rotation & translation
            float costheta = (dT(0,0)+dT(1,1)+dT(2,2) - 1)*0.5f;
            float rot = acos(costheta);

            float dtrans = dT.block<3,1>(0,3).norm();
            if(rot >= THRES_ROTATION_ || dtrans >= THRES_TRANSLATION_) flag_addkeyframe = true;
        }


    }
    if(flag_addkeyframe) {
        std::cout << "       ADD NEW KEYFRAME. keyframe id: ";
        for(std::list<FramePtr>::iterator it = kfs_list_.begin(); it != kfs_list_.end(); ++it){
            std::cout << (*it)->getID() << " ";
        }
        std::cout << "\n";
    }
    else std::cout << "       Do not add keyframe.\n";

    return flag_addkeyframe;
};

const std::list<FramePtr>& Keyframes::getList() const {
    return kfs_list_;
};

int Keyframes::getCurrentNumOfKeyframes() const {
    return kfs_list_.size();
};

int Keyframes::getMaxNumOfKeyframes() const {
    return n_max_keyframes_;
};