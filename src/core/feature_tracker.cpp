#include "core/feature_tracker.h"

FeatureTracker::FeatureTracker()
{
    printf(" - FEATURE_TRACKER is constructed.\n");
};

FeatureTracker::~FeatureTracker()
{
    printf(" - FEATURE_TRACKER is deleted.\n");
};

void FeatureTracker::track(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err,
    PixelVec& pts_track, MaskVec& mask_valid)
{
    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    pts_track.resize(0);
    pts_track.reserve(n_pts);

    std::vector<uchar> status;
    std::vector<float> err;
    int maxLevel = max_pyr_lvl;
    cv::calcOpticalFlowPyrLK(img0, img1, 
        pts0, pts_track, 
        status, err, cv::Size(window_size,window_size), maxLevel);
    
    for(int i = 0; i < n_pts; ++i)
        mask_valid[i] = (mask_valid[i] && status[i] > 0 && err[i] <= thres_err);
    
    // printf(" - FEATURE_TRACKER - 'track()'\n");
};

void scaleRefinement(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0,  uint32_t window_size, float thres_err,
            PixelVec& pts_track, MaskVec& mask_valid)
{
    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

};

void FeatureTracker::trackBidirection(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection,
                PixelVec& pts_track, MaskVec& mask_valid)
{

    float thres_bidirection2 = thres_bidirection*thres_bidirection;

    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    pts_track.resize(0);
    pts_track.reserve(n_pts);

    int maxLevel = max_pyr_lvl;

    // forward tracking
    std::vector<uchar> status_forward;
    std::vector<float> err_forward;
    cv::calcOpticalFlowPyrLK(img0, img1, 
        pts0, pts_track, 
        status_forward, err_forward, cv::Size(window_size,window_size), maxLevel);
    
    // backward tracking
    PixelVec pts0_backward(n_pts);
    std::copy(pts0.begin(), pts0.end(), pts0_backward.begin());
    std::vector<uchar> status_backward;
    std::vector<float> err_backward;
    cv::calcOpticalFlowPyrLK(img1, img0, 
        pts_track, pts0_backward,
        status_backward, err_backward, cv::Size(window_size,window_size), maxLevel-1, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});

    // Check validity.
    for(int i = 0; i < n_pts; ++i)
    {
        Pixel dp = pts0_backward[i]-pts0[i];
        float dist2 = dp.x*dp.x + dp.y*dp.y;

        // border validity
        mask_valid[i] = (mask_valid[i] && pts_track[i].x > 3 && pts_track[i].x < n_cols-3
                                       && pts_track[i].y > 3 && pts_track[i].y < n_rows-3);
        // other ...
        mask_valid[i] = (mask_valid[i] 
            && status_forward[i]
            && status_backward[i]
            && err_forward[i]  <= thres_err
            && err_backward[i] <= thres_err
            && dist2 <= thres_bidirection2
        );
    }
    
    // printf(" - FEATURE_TRACKER - 'trackBidirection()'\n");
};


void FeatureTracker::trackBidirectionWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err, float thres_bidirection, 
            PixelVec& pts_track, MaskVec& mask_valid)
{

    float thres_bidirection2 = thres_bidirection*thres_bidirection;

    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    int maxLevel = max_pyr_lvl;

    PixelVec pts_prior = pts_track;

    // forward tracking
    std::vector<uchar> status_forward;
    std::vector<float> err_forward;
    cv::calcOpticalFlowPyrLK(img0, img1, 
        pts0, pts_track, 
        status_forward, err_forward, cv::Size(window_size, window_size), maxLevel, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});

    // backward tracking
    PixelVec pts0_backward(n_pts);
    std::copy(pts0.begin(), pts0.end(), pts0_backward.begin());
    std::vector<uchar> status_backward;
    std::vector<float> err_backward;
    cv::calcOpticalFlowPyrLK(img1, img0, 
        pts_track, pts0_backward,
        status_backward, err_backward, cv::Size(window_size, window_size), maxLevel, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});

    // Check validity.
    int cnt_inimage      = 0;
    int cnt_forward      = 0;
    int cnt_forward_err  = 0;
    int cnt_backward     = 0;
    int cnt_backward_err = 0;
    int cnt_bidirection  = 0;
    int cnt_prior_works  = 0;

    for(int i = 0; i < n_pts; ++i)
    {
        Pixel dp = pts0_backward[i]-pts0[i];
        float dist2 = dp.x*dp.x + dp.y*dp.y;
        
        bool flag_inimage = pts_track[i].x > 0 && pts_track[i].x < n_cols
                         && pts_track[i].y > 0 && pts_track[i].y < n_rows;
        bool flag_forward      = status_forward[i];
        bool flag_forward_err  = err_forward[i] <= thres_err;
        bool flag_backward     = status_backward[i];
        bool flag_backward_err = err_backward[i] <= thres_err;
        bool flag_bidir        = dist2 <= thres_bidirection2*5;

        cnt_inimage      += flag_inimage;
        cnt_forward      += flag_forward;
        cnt_forward_err  += flag_forward_err;
        cnt_backward     += flag_backward;
        cnt_backward_err += flag_backward_err;
        cnt_bidirection  += flag_bidir;


        mask_valid[i] = (mask_valid[i] 
                        && flag_inimage
                        && flag_forward
                        && flag_forward_err
                        && flag_backward
                        && flag_backward_err
                        && flag_bidir);

        Pixel dp_prior = (pts_prior[i] - pts_track[i]);
        float dist_prior = std::sqrt(dp_prior.x*dp_prior.x + dp_prior.y*dp_prior.y);
        if(dist_prior < 5.0) cnt_prior_works++;
        
    }

    std::cout << "cnt_inimage     : " << cnt_inimage      << " / " << pts_track.size() << std::endl;     
    std::cout << "cnt_forward     : " << cnt_forward      << " / " << pts_track.size() << std::endl;     
    std::cout << "cnt_forward_err : " << cnt_forward_err  << " / " << pts_track.size() << std::endl; 
    std::cout << "cnt_backward    : " << cnt_backward     << " / " << pts_track.size() << std::endl;    
    std::cout << "cnt_backward_err: " << cnt_backward_err << " / " << pts_track.size() << std::endl;
    std::cout << "cnt_bidirection : " << cnt_bidirection  << " / " << pts_track.size() << std::endl; 
    std::cout << "cnt prior works : " << cnt_prior_works  << " / " << pts_track.size() <<std::endl;
    
    // printf(" - FEATURE_TRACKER - 'trackBidirection()'\n");
};

void FeatureTracker::trackWithPrior(const cv::Mat& img0, const cv::Mat& img1, const PixelVec& pts0, uint32_t window_size, uint32_t max_pyr_lvl, float thres_err,
    PixelVec& pts_track, MaskVec& mask_valid)
{
    int n_cols = img0.size().width;
    int n_rows = img0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    // KLT tracking
    int maxLevel = max_pyr_lvl;

    // KLT tracking with prior.
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img0, img1,
        pts0, pts_track,
        status, err, cv::Size(window_size, window_size), maxLevel, {}, cv::OPTFLOW_USE_INITIAL_FLOW, {});
    
    // Check validity.
    for(int i = 0; i < n_pts; ++i) 
    {
        // Border check
        mask_valid[i] = (mask_valid[i] 
                        && status[i] > 0 
                        && pts_track[i].x > 0 && pts_track[i].x < n_cols
                        && pts_track[i].y > 0 && pts_track[i].y < n_rows);
        // Error check
        mask_valid[i] = (mask_valid[i] && err[i]  <= thres_err);

        // ZNCC check
        // if(mask_valid[i]){
        //     float ncc_track = image_processing::calcZNCC(img0,img1,pts0[i],pts_track[i],15);
        //     mask_valid[i] = ncc_track > 0.5f;
        // }
    }
    
    // printf(" - FEATURE_TRACKER - 'trackWithPrior()'\n");
};

void FeatureTracker::calcPrior(const PixelVec& pts0, const PointVec& Xw, const PoseSE3& Tw1, const Eigen::Matrix3f& K,
    PixelVec& pts1_prior) 
{
    int n_pts = Xw.size();
    pts1_prior.resize((int)(pts0.size()));
    std::copy(pts0.begin(),pts0.end(), pts1_prior.begin());

    Eigen::Matrix4f T1w = Tw1.inverse();
    PointVec X1;
    X1.reserve(n_pts);
    for(auto Xw_ : Xw) X1.emplace_back((T1w.block(0,0,3,3)*Xw_ + T1w.block(0,3,3,1)));

    for(int i = 0; i < n_pts; ++i)
    {   
        Point X = X1[i];
        int mask_haveprior = X.norm() > 0;
        if(mask_haveprior){
            Pixel pt_tmp;
            pt_tmp.x = K(0,0)*X(0)/X(2) + K(0,2);
            pt_tmp.y = K(1,1)*X(1)/X(2) + K(1,2); // projection
            pts1_prior[i] = pt_tmp ;
        }
        // else pts1_prior[i] = pts0[i];
    }
};


void FeatureTracker::refineScale(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& dimg1_u, const cv::Mat& dimg1_v, const PixelVec& pts0, const float& scale_init,
        PixelVec& pts_track, MaskVec& mask_valid)
{
    std::cout << "refine do\n";
    if(pts_track.size() != pts0.size() ) throw std::runtime_error("pts_track.size() != pts0.size()");
    

    cv::Mat I0, I1;
    img0.convertTo(I0, CV_32FC1);
    img1.convertTo(I1, CV_32FC1);

    cv::Mat dI1u, dI1v;
    dI1u = dimg1_u;
    dI1v = dimg1_v;
    // std::cout << image_processing::type2str(dI1u) << std::endl;

    int n_cols = I0.size().width;
    int n_rows = I0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);

    int win_sz     = 10;
    int win_len    = 2*win_sz+1;
    int win_len_sq = win_len*win_len;

    int   MAX_ITER  = 150;
    float EPS_SCALE = 1e-4;
    float EPS_PIXEL = 1e-3;
    float EPS_TOTAL = 1e-5;

    // Generate patch
    PixelVec patt(win_len_sq);
    int ind = 0;
    for(int j = 0; j < win_len; ++j) {
        for(int i = !(j & 0x01); i < win_len; i += 2) {
            patt[ind].x = (float)(i-win_sz);
            patt[ind].y = (float)(j-win_sz);
            ++ind;
        }
    }
    win_len_sq = ind;

    // containers
    std::vector<float> I0_patt(win_len_sq);
    std::vector<float> I1_patt(win_len_sq);
    std::vector<float> dI1u_patt(win_len_sq);
    std::vector<float> dI1v_patt(win_len_sq);

    // temporal container
    PixelVec pts_warp(win_len_sq);
    MaskVec  mask_warp_I0(win_len_sq);
    MaskVec  mask_warp_I1(win_len_sq);
    
    // Iteratively update for each point.
    for(int i = 0; i < n_pts; ++i) 
    { 
        if(!mask_valid[i]) 
            continue;

        // calculate prior values.
        Eigen::MatrixXf theta(3,1);
        theta(0,0) = 0.0;
        theta(1,0) = 0.0;
        theta(2,0) = 1.15f; // initial scale. We assume increasing scale.

        for(int j = 0; j < win_len_sq; ++j)
            pts_warp[j] = patt[j] + pts0[i];

        // interpolate data
        image_processing::interpImage(I0, pts_warp, I0_patt, mask_warp_I0);
        
        float err_curr = 0;
        float err_prev = 1e22;

        // Iterations.
        for(int iter = 0; iter < MAX_ITER; ++iter) 
        {
            Pixel pt_track_tmp;
            pt_track_tmp.x = theta(0,0) + pts_track[i].x;
            pt_track_tmp.y = theta(1,0) + pts_track[i].y;

            // warp patch points
            for(int j = 0; j < win_len_sq; j++)
                pts_warp[j] = patt[j]*theta(2,0) + pt_track_tmp;
            
            // Generate patches.
            // image_processing::interpImage(I1,   pts_warp,   I1_patt, mask_warp);
            // image_processing::interpImage(dI1u, pts_warp, dI1u_patt, mask_warp);
            // image_processing::interpImage(dI1v, pts_warp, dI1v_patt, mask_warp);
            image_processing::interpImage3(I1, dI1u, dI1v, pts_warp,
                I1_patt, dI1u_patt, dI1v_patt, mask_warp_I1);

            // calculate jacobian & residual
            Eigen::MatrixXf J(win_len_sq, 3); // R^{N x 3}
            Eigen::MatrixXf r(win_len_sq, 1); // R^{N x 1}
            int idx_tmp = 0;
            for(int j = 0; j < win_len_sq; ++j) 
            {
                if(mask_warp_I0[j] && mask_warp_I1[j]) 
                {
                    J(idx_tmp,0) = dI1u_patt[j];
                    J(idx_tmp,1) = dI1v_patt[j];
                    J(idx_tmp,2) = dI1u_patt[j]*patt[j].x + dI1u_patt[j]*patt[j].y;
                    
                    r(idx_tmp,0) = I1_patt[j] - I0_patt[j];
                    ++idx_tmp;
                }
            }
            
            // calculate Hessian and HinvJt
            Eigen::MatrixXf H(3,3);                // equal to J^t*J, R^{3 x 3}
            Eigen::MatrixXf HinvJt(3,win_len_sq); // R^{3 x N}

            J      = J.block(0,0,idx_tmp,3);
            r      = r.block(0,0,idx_tmp,1);
            H      = J.transpose()*J;
            HinvJt = H.inverse()*J.transpose();
            HinvJt = HinvJt;

            Eigen::MatrixXf delta_theta(3,1);
            delta_theta = -HinvJt*r;
            theta = theta + delta_theta;

            err_curr = r.norm()/r.size();

            // breaking point.
            if( abs(err_prev-err_curr) <= EPS_TOTAL || sqrt(delta_theta(0,0)*delta_theta(0,0)+delta_theta(1,0)*delta_theta(1,0)) < EPS_PIXEL)
            {
                // std::cout << " e:" << err_curr <<", itr:" << iter << "\n";
                // std::cout <<"update  px: " << theta(0,0)<< ", " << theta(1,0) << ", sc:" << theta(2,0) << std::endl;
                break;
            }
            err_prev = err_curr;
        }

        // push results
        if(err_curr < 20 && theta(2,0) > 0.8 && theta(2,0) < 1.3) 
        {
            pts_track[i].x += theta(0,0);
            pts_track[i].y += theta(1,0);
            mask_valid[i] = true;
        }
        else 
            mask_valid[i] = false; // not update
    }

};

void FeatureTracker::trackWithScale(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& dI0u, const cv::Mat& dI0v, const PixelVec& pts0, const std::vector<float>& scale_est,
            PixelVec& pts_track, MaskVec& mask_valid)
{
    /* 
        <Cost function>
            argmin_{t} SUM_{j} rij^2 where rij = I1(w(p_i, ppat_j, t)) - I0(p_i, ppat_j)

        <How to solve>
        --> Inverse Compositional Method

        <Linearization>
            rij(t-dt) = I1[ w(p_i, ppat_j, t) ] - I0[ w(p_i, ppat_j, dt) ]

        <Update>
            t_new <-- t - dt;

        drij_ddt = -dI0/dp * dw/dt = -[du0,dv0]*[1,0;0,1] = -[du0,dv0];

        J = [J1,J2] = -[dIu0, dIv0]
        r = I1 - I0
        
        (J.'*J)*dt +J.'*r = 0 --> dt = -(J.'*J)^-1*J.'r
        Thus, A = Jt*J , b = -Jt*r       

        A11 += du0*du0
        A12 += du0*dv0
        A22 += dv0*dv0
        b1  += du0*r
        b2  += dv0*r
                
        A*dt = b --> dt = A^-1*b.
        [ A11,A12 ] * [ dtu ]  = [ b1 ]
        [ A12,A22 ]   [ dtv ]    [ b2 ]
        
        D = A11*A22 - A12*A12 // related to the cornerness
        invD = 1.0/D
        [ -A22  A12 ]
        [  A12 -A11 ]
        dtu = (-A22*b1 + A12*b2 ) * invD
        dtv = ( A12*b1 - A11*b2 ) * invD
        
        Jt*J = [sum(du0*du0), sum(du0*dv0); sum(du0*dv0), sum(dv0*dv0)]
        Jt*r = [J1*r; J2*r]; 

    */
    std::cout << "TrackWithScale do\n";
    if(pts_track.size() != pts0.size() ) throw std::runtime_error("pts_track.size() != pts0.size()");
    
    int half_win_sz = 10;
    int win_len     = 2*half_win_sz+1;
    int n_elem      = win_len*win_len;

    int   MAX_ITER  = 150;
    float EPS_PIXEL = 1e-3;
    float EPS_TOTAL = 1e-5;


    cv::Mat I0, I1;
    img0.convertTo(I0, CV_32FC1);
    img1.convertTo(I1, CV_32FC1);

    int n_cols = I0.size().width;
    int n_rows = I0.size().height;

    int n_pts = pts0.size();
    mask_valid.resize(n_pts, true);


    // Generate patch
    PixelVec patt(n_elem);
    PixelVec patt_s(n_elem);
    int ind = 0;
    for(int v = 0; v < win_len; ++v) {
        for(int u = !(v & 0x01); u < win_len; u += 2) {
            patt[ind].x = (float)(u - half_win_sz);
            patt[ind].y = (float)(v - half_win_sz);
            ++ind;
        }
    }
    n_elem = ind;
    std::cout << "n_elem: " << n_elem << std::endl;

    // containers
    std::vector<float> I0_patt(n_elem);
    std::vector<float> dI0u_patt(n_elem);
    std::vector<float> dI0v_patt(n_elem);
    std::vector<float> I1_patt(n_elem);

    // temporal container
    PixelVec pts_warp(n_elem);
    MaskVec  mask_I0(n_elem);
    MaskVec  mask_I1(n_elem);
    
    // Iteratively update for each point.
    for(int i = 0; i < n_pts; ++i) // for each point
    { 
        // std::cout <<i <<"-th point\n";
        // If invalid point, skip.
        if(!mask_valid.at(i)) 
            continue;

        const Pixel& pt0   = pts0.at(i);
        const Pixel& pt1   = pts_track.at(i);
        const float& scale = scale_est.at(i);
        // std::cout << "scale: " << scale << std::endl;

        // Make scaled patch , Calculate patch points
        for(int j = 0; j < n_elem; ++j)
        {
            patt_s[j]   = patt[j]*scale;
            pts_warp[j] = pt0 + patt[j];
            // std::cout << j << "-th patt: " << pts_warp[j] << std::endl;
        }

        // interpolate data
        float ax   = pt0.x - floor(pt0.x);
        float ay   = pt0.y - floor(pt0.y);
        float axay = ax*ay;
        image_processing::interpImage3SameRatio(I0, dI0u, dI0v, pts_warp, 
            ax,ay,axay,
            I0_patt, dI0u_patt, dI0v_patt, mask_I0);

        float A11 = 0, A12 = 0, A22 = 0;
        for(int j = 0; j < n_elem; ++j)
        {   
            if(mask_I0[j])
            {   
                const float& du0 = dI0u_patt[j];
                const float& dv0 = dI0v_patt[j];

                A11 += du0*du0;
                A12 += du0*dv0;
                A22 += dv0*dv0;
            }
        }            
        float D = A11*A22 - A12*A12;
        float invD = 1.0/D;  

        // Iterations.
        float err_curr = 0;
        float err_prev = 1e12;
        Pixel t = pt1 - pt0; // initialize dp
        Pixel pt_update;
        for(int iter = 0; iter < MAX_ITER; ++iter) 
        {
            // updated point
            pt_update = pt0 + t;

            // calculate ratios
            ax   = pt_update.x - floor(pt_update.x);
            ay   = pt_update.y - floor(pt_update.y);
            axay = ax*ay;

            // warp patch points
            for(int j = 0; j < n_elem; ++j)
                pts_warp[j] = pt_update + patt_s[j];

            // interpolate data
            image_processing::interpImageSameRatio(I1, pts_warp, 
                ax, ay, axay, 
                I1_patt, mask_I1);

            // Calculate Hessian, Jacobian and residual .
            float b1 = 0, b2 = 0;
            int cnt_valid = 0;       
            err_curr = 0;
            for(int j = 0; j < n_elem; ++j)
            {   
                if(mask_I0[j] && mask_I1[j])
                {   
                    const float& du0 = dI0u_patt[j];
                    const float& dv0 = dI0v_patt[j];
                    float r = I1_patt[j] - I0_patt[j];

                    b1 += du0*r;
                    b2 += dv0*r;
                    err_curr += r*r;

                    ++cnt_valid;
                }
            }

            // Solve update step
            float dtu = (-A22*b1 + A12*b2 ) * invD;
            float dtv = ( A12*b1 - A11*b2 ) * invD;

            // std::cout << dtu <<"," << dtv << std::endl;

            t.x += dtu;
            t.y += dtv;


            // Evaluate current update. (breaking point)
            err_curr /= (float)cnt_valid;
            err_curr = sqrt(err_curr);
            float stepsize = (dtu*dtu + dtv*dtv);
            std::cout << iter << "-th iter: " << t << "/ err: " << err_curr << ", stepsize: "<< stepsize << std::endl;
            if( (iter > 1 && abs(err_prev - err_curr) / err_prev <= 1e-5)
            || (dtu*dtu + dtv*dtv) < EPS_PIXEL)
            {
                break;
            }

            err_prev = err_curr;
        } // END iter

        // push results
        if(err_curr <= 30) 
        {
            pts_track[i]  = pt0 + t;
            mask_valid[i] = true;
        }
        else
            mask_valid[i] = false; // not update.
        
    } // END i-th pts
};
