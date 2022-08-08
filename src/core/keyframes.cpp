#include "core/keyframes.h"

Keyframes::Keyframes()
: THRES_OVERLAP_FEATURE_RATIO_(0.7f),
THRES_ROTATION_(3.0f*D2R),
THRES_TRANSLATION_(2.0f),
n_max_keyframes_(7)
{ 

};

void Keyframes::setMaxKeyframes(int max_kf){
    n_max_keyframes_ = max_kf;
};

void Keyframes::addNewKeyframe(const FramePtr& frame){
    if(keyframes_.size() == n_max_keyframes_) {
        all_keyframes_.push_back(keyframes_.front());
        keyframes_.pop_front();
    }
    keyframes_.push_back(frame);

};

bool Keyframes::checkUpdateRule(const FramePtr& frame_curr){
    bool flag_addkeyframe = false;
    if(keyframes_.empty()) flag_addkeyframe = true;

    if(!flag_addkeyframe){
        
        std::vector<float> tracking_ratios;
        for(std::list<FramePtr>::iterator it = keyframes_.begin(); it != keyframes_.end(); ++it){
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
        const PoseSE3& Twk_last = keyframes_.back()->getPose();
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
        for(std::list<FramePtr>::iterator it = keyframes_.begin(); it != keyframes_.end(); ++it){
            std::cout << (*it)->getID() << " ";
        }
        std::cout << "\n";
    }
    else std::cout << "       Do not add keyframe.\n";

    return flag_addkeyframe;
};

void Keyframes::localBundleAdjustment(){
    std::cout << "Local Bundle adjustment\n";
    int n_keyframes = keyframes_.size();

    std::vector<std::set<LandmarkPtr>> landmark_sets;
    std::set<LandmarkPtr> landmark_set_all;
    
    for(std::list<FramePtr>::iterator it = keyframes_.begin(); it != keyframes_.end(); ++it){
        landmark_sets.push_back(std::set<LandmarkPtr>());
        
        for(auto lm : (*it)->getRelatedLandmarkPtr()){
            landmark_sets.back().insert(lm);
            landmark_set_all.insert(lm);
        }
    }

    std::cout << "landmark set size: ";
    for(auto lmset : landmark_sets){
        std::cout << lmset.size() << " " ;
    }
    std::cout << std::endl;
    std::cout << "distinguished landmarks: " << landmark_set_all.size() << std::endl;

    //존재 하는 원소 찾기    
    std::set<LandmarkPtr>::iterator iter;
    iter = landmark_set_all.find(keyframes_.back()->getRelatedLandmarkPtr().back());
    if(iter != landmark_set_all.end()) std::cout << *iter << ":존재 "<< std::endl;
    else std::cout << "존재하지않음.\n";
    
        
};