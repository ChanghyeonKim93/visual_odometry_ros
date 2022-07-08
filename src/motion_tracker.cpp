#include "motion_tracker.h"

MotionTracker::MotionTracker(){

};

MotionTracker::~MotionTracker(){
    
};

cv::Mat MotionTracker::trackCurrentImage(const cv::Mat& img, const double& timestamp){
    // get motion...
    this->img_current_ = img;
    if(img_current_.channels() != 1){
        throw std::runtime_error("grabImageMonocular - Image is not grayscale image.");
    }

    if(track_state_ == TrackingState::NOT_INITIALIZED || track_state_ == TrackingState::NO_IMAGES_YET){
        // Initialize the current frame
    }
    else {
        
    }
    
    // track the input image 
    track();
    
    // return current_frame_.Tcw.clone();
};  

void MotionTracker::track(){
    
    if(track_state_ = TrackingState::NOT_INITIALIZED){
         monocularInitialization();

        // not yet initialized?
        if(track_state_ != TrackingState::OK) return;
    }
    else{
        // Already initialized. track.

        bool flag_OK;

        // Initial camera pose estimation using motion model
        // Local Mapping is activated. This is the normal behaviour, unless
        // you explicitly activate the "only tracking" mode.

        if(track_state_== TrackingState::OK) {
            // Local Mapping might have changed some MapPoints tracked in last frame
            CheckReplacedInLastFrame();

            if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
            {
                flag_OK = TrackReferenceKeyFrame();
            }
            else
            {
                flag_OK = TrackWithMotionModel();
                if(!flag_OK)
                    flag_OK = TrackReferenceKeyFrame();
            }
        }
        else
        {
            // bOK = Relocalization(); //
        }

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(track_state_ == TrackingState::OK)
            flag_OK = TrackLocalMap();

        if(flag_OK) track_state_ = TrackingState::OK;
        else track_state_ = TrackingState::LOST;

        // If tracking were good, check if we insert a keyframe
        if(flag_OK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
            else
                mVelocity = cv::Mat();


            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }

                // Delete temporal MapPoints
                for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
                {
                    MapPoint* pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                if(NeedNewKeyFrame())
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for(int i=0; i<mCurrentFrame.N;i++)
                {
                    if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                }
            }

        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }    
};

void MotionTracker::monocularInitialization(){

};