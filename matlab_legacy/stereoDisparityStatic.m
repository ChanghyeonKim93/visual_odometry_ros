function frame_k = stereoDisparityStatic(frame_k)
% 이 함수에서 업데이트 해야하는 것.
% (1) 대상이 되는 점의 깊이값, (2) 3차원 좌표 생성, (3) 역수깊이, (4) 깊이 표준편

% intrinsic 및 extrinsic load
global data_info
n_cols = data_info.intrin.n_cols.left;
n_rows = data_info.intrin.n_rows.left;
K    = data_info.rectify.K; % rectification 된 K
Kinv = data_info.rectify.Kinv;% rectification 된 Kinv
g_lr = data_info.extrin.g_nlnr; % rectify된 left right 자세 값.

% 이미지 로드
img_l = frame_k.left.img;
img_r = frame_k.right.img;

% 복원 대상 점 & 그 점들의 그래디언트
pts_l  = frame_k.left.pts;
grad_l = frame_k.left.grad;


n_pts = length(pts_l);
scores_best = -ones(1,n_pts);
flag_valid  = zeros(1,n_pts);
invd_save = -ones(1,n_pts);
std_invd_save = -ones(1,n_pts);
idx_alive_save = -ones(1,n_pts);

% 파라미터 설정
flag_draw      = false;
flag_refine    = true;

win_sz      = 8; % half length of window.
fat_sz      = 1; % half thicKess of window.

angle_thres  = 60; % 이 각도보다 grad 와 epiline의 각도가 크면 복원 x
cosd_thres = cosd(angle_thres);
thres_zncc1  = 0.95;
thres_zncc2  = 0.9;
thres_multipeak = 5;

eps_edge     = 0.5; % 0.5 [px]
eps_epi      = 1; % 1 [px]


% minimum & maximum disparity 값. 해당 disparity 범위 안에서만 서칭.
% baseline 대비, 0.4 m ~ 15 m 범위에서 가능한 disparity 값의 범위를 계산해둔다.
baseline  = norm(g_lr(1:3,4)); % [m]
focal     = K(1,1);
d_min     = 0.3; % 0.5 [m]
d_max     = 25;  % 20  [m]
bf        = baseline*focal;
bfinv     = 1/bf;
max_disp  = bf/d_min;
min_disp  = bf/d_max;

%% 본격적인 깊이 복원
% 패치 선계산
u_patch = -win_sz:win_sz;
v_patch = -fat_sz:fat_sz;
[X,Y] = meshgrid(-win_sz:win_sz, -fat_sz:fat_sz);
pts_patch = [X(:)';Y(:)'];

u_min = -1; % 최소 서칭범위
u_max = -1; % 최대 서칭범위

for i = 1:n_pts
   % 현재 i 번째 점의 데이터를 추출한다.
   u_l  = pts_l(1,i);
   v_l  = pts_l(2,i);
   
   du_l = grad_l(1,i);
   dv_l = grad_l(2,i);
   dmag_l = sqrt(du_l^2+dv_l^2);
   du_l = du_l / dmag_l;
   dv_l = dv_l / dmag_l;
   
   % gradient direction이 epipolar line와 이루는 각도가 70도 이내만 복원.
   if((abs(du_l) >= cosd_thres) && ...
         (u_l > win_sz + 1) && (u_l < n_cols - win_sz) &&...
         (v_l > fat_sz + 1) && (v_l < n_rows - fat_sz))
      
      % 패치 추출
      patch_l = interpImage(img_l, pts_patch + [u_l;v_l]);
      
      % 만약, 깊이 값이 이미 있다면, (이전에 워핑된) +-2*std 범위에서 disparity 서칭 범위 지정
      if(frame_k.left.invd(i) > 0)
         invd_temp = frame_k.left.invd(i);
         std_temp  = frame_k.left.std_invd(i);
         u_min = floor(u_l - bf*(invd_temp + 3*std_temp));
         u_max =  ceil(u_l - bf*(invd_temp - 3*std_temp));
      else
         % 깊이값이 없다면, 전체 범위에서 탐색.
         u_min = floor(u_l - max_disp);
         u_max =  ceil(u_l - min_disp);
      end
      % 서칭범위를 이미지 내부로 제한해준다.
      if(u_min < win_sz + 1)
         u_min = win_sz + 1;
      end
      if(u_max > n_cols - win_sz -1)
         u_max = n_cols - win_sz -1;
      end
      
      % 매칭쌍을 찾는다
      scores       = -ones(1,u_max-u_min+1); % 서칭하는 곳의 모든 score저장.
      score_best   = -1e5;
      u_over = []; % thres2를 (0.8)을 넘은 점들을 모아둠.
      u_best = -1; % 매칭된 점의 위치. (subpixel)
      
      % 범위를 순회하며 서칭.
      ind_best = -1;
      for j = 1:(u_max - u_min + 1)
         u_r = j - 1 + u_min;
         patch_r = img_r(v_l + v_patch, u_r + u_patch);
         scores(j)  = calcZNCC(patch_l(:), patch_r(:));
         if(scores(j) > thres_zncc2)
            u_over = [u_over, u_r];
         end
         
         % 만약, 현재 ZNCC가 이전 값 보다 높으면 바꾼다.
         if(scores(j) > score_best)
            score_best = scores(j);
            u_best = u_r;
            ind_best = j;
         end
      end
      
      % 최고 점수가 threshold를 넘고, 모든 u_over가 6 pixel 이내이고,
      % 범위 가장자리가 아닌 경우에만 신뢰.
      flag_valid(i) = 1;
      for j = 1:length(u_over) % 하나라도 6 px 이상 벗어나있다면 기각.
         if(abs(u_best - u_over(j)) > thres_multipeak)
            flag_valid(i) = 0;
            break;
         end
      end
      if(flag_valid(i) && (score_best > thres_zncc1) && (u_best > u_min + 2) && (u_best < u_max - 2))
         % 최고 점수를 저장한다.
         scores_best(i) = score_best;
         % score history 이용해서 2차함수 모델로 subpixel refinement 수행.
         s1 = scores(ind_best-1); % 왼쪽 값.
         s2 = scores(ind_best); % 중심 값.
         s3 = scores(ind_best+1); % 우측 값.
         u_refine    = u_best - (s3-s1)/(s3+s1-2*s2)*0.5; % refine된 좌표 값.
         dispar_temp = u_l - u_refine;
         
         % inverse depth를 계산한다.
         invd_save(i) = dispar_temp*bfinv;
         % standard deviation을 계산한다.
         std_invd_save(i) = 1/abs(du_l)*sqrt(eps_edge^2 + eps_epi^2*dv_l^2)*bfinv;
      else
         flag_valid(i) = 0;
      end
   end
end

% 결과 업데이트
for k = 1:n_pts
   if(flag_valid(k) > 0)
      invd_prev = frame_k.left.invd(k);
      std_prev = frame_k.left.std_invd(k);
      
      invd_curr = invd_save(k);
      std_curr  = std_invd_save(k);
      if(invd_prev > 0) % (1) inverse depth가 있던 경우.
         % 퓨전.
         updateDF(frame_k.left.df.a(k)
         
         invd_fusion = 1/(std_prev^2+std_curr^2)*(invd_prev*std_curr^2 + invd_curr*std_prev^2);
         std_fusion   = sqrt( (std_prev^2*std_curr^2)/(std_prev^2+std_curr^2) );
         
         % inverse depth를 계산한다.
         frame_k.left.invd(k)     = invd_fusion;
         % standard deviation을 계산한다.
         frame_k.left.std_invd(k) = std_fusion;
         % pts3d를 계산한다.
         frame_k.left.pts3d(:,k) = Kinv*[frame_k.left.pts(:,k);1]/invd_fusion;
         frame_k.left.is_recon(k) = 1;
         
      else
         % inverse depth를 계산한다.
         frame_k.left.invd(k)     = invd_curr;
         % standard deviation을 계산한다.
         frame_k.left.std_invd(k) = std_curr;
         % pts3d를 계산한다.
         frame_k.left.pts3d(:,k)  = Kinv*[frame_k.left.pts(:,k);1]/invd_curr;
         frame_k.left.is_recon(k) = 1;
      end      
   end
end

end