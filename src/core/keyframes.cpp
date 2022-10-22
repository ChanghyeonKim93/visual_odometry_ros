#include "core/keyframes.h"

Keyframes::Keyframes()
: THRES_OVERLAP_FEATURE_RATIO_(0.7),
THRES_ROTATION_(3.0f*D2R), THRES_TRANSLATION_(1.0f),
N_MAX_KEYFRAMES_IN_WINDOW_(7)
{

};

void Keyframes::setMaxKeyframes(int max_kf)
{
    N_MAX_KEYFRAMES_IN_WINDOW_ = max_kf;
};

void Keyframes::addNewKeyframe(const FramePtr& frame)
{
    frame->makeThisKeyframe(); // 새 keyframe이 됨을 표시. (추가시에는 당연히 keyframe window로 들어옴)
    all_keyframes_.push_back(frame); // keyframe 저장.

    if(kfs_list_.size() == N_MAX_KEYFRAMES_IN_WINDOW_) 
    {
        kfs_list_.front()->outOfKeyframeWindow(); // keyframe window에서 제거됨을 표시.
        kfs_list_.pop_front(); // keyframe window에서 제거.
    }
    kfs_list_.push_back(frame); // 새 keyframe을 추가. (sliding window)
    
    // 새로 추가된 keyframe과 관련된 landmark 정보를 업데이트.
    for(const auto& lm : frame->getRelatedLandmarkPtr())
        lm->addObservationAndRelatedKeyframe(lm->getObservations().back(), frame);
};

bool Keyframes::checkUpdateRule(const FramePtr& frame_curr)
{
    bool flag_addkeyframe = false;

    if(kfs_list_.empty()) 
        flag_addkeyframe = true; // Keyframe이 없으면 무조건 추가한다.

    if(!flag_addkeyframe)
    {   
        // 1) Check overlapping ratio
        // 키프레임이 이미 있는 경우에 대한 고려임.
        // 각 키프레임에 얼마나 많은 landmarks가 추적되었는지 계산하기.
        std::vector<float> tracking_ratios;
        for(std::list<FramePtr>::iterator it = kfs_list_.begin(); it != kfs_list_.end(); ++it)
        {
            // 과거 시점의 키프레임
            const FramePtr& kf = *it;

            // calculate tracking ratio
            int cnt_tracked   = 0; // 과거 특정 키프레임에서 보였던 landmark 중 현재 프레임으로 추적 된 개수
            int cnt_total_lms = 0; // 과거 특정 키프레임에서 보였던 모든 landmark갯수
            int cnt_alive_lms = 0; // 과거 특정 키프레임에서 보였던 landmark중 살아있는 것 개수
            for(const auto& lm : kf->getRelatedLandmarkPtr())
            {
                ++cnt_total_lms;

                if(lm->getRelatedFramePtr().back() == frame_curr) 
                    ++cnt_tracked; // kf의 landmark가 현재 프레임으로 추적 되었는지.
            }
            float tracking_ratio = (float)cnt_tracked/(float)cnt_total_lms;
            tracking_ratios.push_back(tracking_ratio);
        }
        
        std::cout << "       Tracking ratios : ";
        for(const auto& r : tracking_ratios)
        {
            std::cout << (int)(r*100.f) << "% ";
        }
        std::cout << "\n";
        
        if(tracking_ratios.back() <= THRES_OVERLAP_FEATURE_RATIO_) 
            flag_addkeyframe = true;
        
        // 2) Check rotation & translation
        const PoseSE3& Tkw_last = kfs_list_.back()->getPoseInv();
        const PoseSE3& Twc = frame_curr->getPose();
        PoseSE3 dT = Tkw_last * Twc;
        if(!flag_addkeyframe)
        {
            // rotation & translation
            float costheta = (dT(0,0)+dT(1,1)+dT(2,2) - 1.0f)*0.5f;
            if(costheta >=  0.999999) costheta =  0.999999;
            if(costheta <= -0.999999) costheta = -0.999999;
            float rot = acos(costheta);

            float dtrans = dT.block<3,1>(0,3).norm();
            if(rot >= THRES_ROTATION_ || dtrans >= THRES_TRANSLATION_) 
                flag_addkeyframe = true;
        }
    }

    // Visualization if new keyframe is needed.
    if( flag_addkeyframe ) 
    {
        std::cout << "       ADD NEW KEYFRAME. keyframe id: ";
        for(std::list<FramePtr>::iterator it = kfs_list_.begin(); it != kfs_list_.end(); ++it)
        {
            std::cout << (*it)->getID() << " ";
        }
        std::cout << "\n";
    }
    else // No need of new keyframe
        std::cout << "       Don't add keyframe.\n";

    return flag_addkeyframe;
};

const std::list<FramePtr>& Keyframes::getList() const {
    return kfs_list_;
};

int Keyframes::getCurrentNumOfKeyframes() const {
    return kfs_list_.size();
};

int Keyframes::getMaxNumOfKeyframes() const {
    return N_MAX_KEYFRAMES_IN_WINDOW_;
};