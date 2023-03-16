function frame_k = stereoDisparityTemporal(frame_k, frame_c, g_ck)
% 이 함수에서 업데이트 해야하는 것.
% left 이미지끼리 비교해서 temporal update 를 수행한다.

% intrinsic 및 extrinsic load
global data_info
n_cols = data_info.intrin.n_cols.left;
n_rows = data_info.intrin.n_rows.left;
K    = data_info.rectify.K; % rectification 된 K
Kinv = data_info.rectify.Kinv;% rectification 된 Kinv

R_ck = g_ck(1:3,1:3);
t_ck = g_ck(1:3,4);
t_kc = -R_ck'*t_ck;

% Fundamental 행렬 & Essential 행렬
F_ck = Kinv'*hat(t_ck)*R_ck*Kinv;

% key & cur 이미지 로드
img_k = frame_k.left.img;
img_c = frame_c.left.img;

% 복원 대상 점 & 그 점들의 그래디언트
pts_k  = frame_k.left.pts;
grad_k = frame_k.left.grad;

n_pts       = length(pts_k);
scores_best = -ones(1,n_pts);
flag_valid  = zeros(1,n_pts);

invd_save   = -ones(1,n_pts);
std_invd_save = -ones(1,n_pts);

% 파라미터 설정
win_sz      = 8; % half length of window.
fat_sz      = 3; % half thicKess of window.

angle_thres  = 60; % 이 각도보다 grad 와 epiline의 각도가 크면 복원 x
cosd_thres   = cosd(angle_thres);
thres_zncc1  = 0.95; % NCC가 이 값을 넘지 않으면 복원 X
thres_zncc2  = 0.75; % NCC가 이 값을 넘는 녀석이 여러개가 있으면 체크해본다.
thres_multipeak = 4;

eps_edge     = 0.5; % 엣지 에러 std: 0.5 [px]
eps_epi      = 1;   % epipolar line 에러 std: 1 [px]

% minimum & maximum disparity 값. 해당 disparity 범위 안에서만 서칭.
% baseline 대비, 0.4 m ~ 15 m 범위에서 가능한 disparity 값의 범위를 계산해둔다.
baseline  = norm(g_ck(1:2,4)); % [m] Z 방향으로 가는건 무시?
focal     = K(1,1);

bf = baseline*focal;
bfinv = 1/bf;

d_min_default = 0.1; % 0.1 [m] % 0.1m 부터,
d_max_default = 40;  % 30  [m] % 40m 까지 보도록.

fprintf('max disp: %0.0f / min disp: %0.0f\n', bf/d_min_default, bf/d_max_default);

% 패치 템플릿!
[X,Y] = meshgrid(-win_sz:win_sz, -fat_sz:fat_sz);
pts_patch_template = [X(:)';Y(:)'];

%% 본격적인 깊이 복원
for i = 1:n_pts
   % 현재 i 번째 점의 데이터를 추출한다.
   pt_k = pts_k(:,i);
   du_k = grad_k(1,i);
   dv_k = grad_k(2,i);
   dmag_k = sqrt(du_k^2+dv_k^2);
   
   du_k = du_k / dmag_k;
   dv_k = dv_k / dmag_k;
   
   % cur 에 대한 epipolar line 방향의 vector와 u절편을 찾는다. l_c,l0_c;
   coefficient_c = F_ck*[pt_k;1];
   l_c  = [coefficient_c(2),-coefficient_c(1)]';
   l_c  = l_c/norm(l_c); % normalized vector.
   pt_c_arbi = warpPts(pt_k, 10, g_ck);
   % key 에 대한 epipolar line 방향의 vector를 u찾는다. l_k, l0_k;
   coefficient_k = [pt_c_arbi;1]'*F_ck;
   l_k  = [coefficient_k(2),-coefficient_k(1)]';
   l_k  = l_k/norm(l_k);
   l0_k = [-coefficient_k(3)/coefficient_k(1),0]';
   
   % 두 선들의 방향을 맞춰준다. (내적이 양수이면 됨)
   if(dot(l_c,l_k) < 0)
      l_c = -l_c;
   end
   
   cos_th = dot(l_k,[du_k,dv_k]);
   sin_th = sqrt(1-cos_th^2);
   
   % gradient 방향이 key epi line와 이루는 각도가 60도 이내만 복원.
   if( (abs(cos_th) >= cosd_thres) && ...
         (pt_k(1) > win_sz + 1) && (pt_k(1) < n_cols - win_sz - 1) &&...
         (pt_k(2) > win_sz + 1) && (pt_k(2) < n_rows - win_sz - 1))
      
      % 일단 valid 만든다.
      flag_valid(i) = 1;
      % 패치를 만든다.
      pts_k_patch = calcEpiPatchTemplate(pt_k, l_k, pts_patch_template);
      patch_k     = interpImage(img_k, pts_k_patch);
      
      % 만약, 깊이 값이 이미 있다면, (이전에 워핑된) +-2*std 범위에서 disparity 서칭 범위 지정
      if(frame_k.left.invd(i) > 0)% 깊이값이 없다면, 최대 range에서
         invd_temp = frame_k.left.invd(i);
         std_temp  = frame_k.left.std_invd(i);
         d_min = 1/(invd_temp + 2*std_temp);
         d_max = 1/(invd_temp - 2*std_temp);
      else
         d_min = d_min_default;
         d_max = d_max_default;
      end
      % cur 이미지보다는 앞인지 확인해준다.
      if(d_min < t_kc(3) + d_min_default)
         d_min = t_kc(3) + d_min_default;
      end
      % 지금의 범위 밖에 있는 깊이라면 기각
      if(d_max > d_max_default)
         flag_valid(i) = 0;
      end
      % 현재 알고있는 depth의 min max 값을 이용해서 keyframe에서 current frame으로 워핑해준다.
      pt_c_start = warpPts(pt_k, d_max, g_ck);
      pt_c_end   = warpPts(pt_k, d_min, g_ck);
      
      % 바운더리에 pt_end - pt_start 선분이 걸치는지 확인해본다.
      % 서칭범위를 이미지 내부 / 최대 & 최소 깊이 정보를 고려하여 범위를 좁혀준다.
%       [pt_c_start, pt_c_end, n_search] = testBoundary(pt_c_start, pt_c_end, l_c, n_rows, n_cols);
      
      if( dot(l_c,pt_c_end-pt_c_start) < 0 ) % l_c와 방향 같게 만들어.
         pt_temp = pt_c_start;
         pt_c_start = pt_c_end;
         pt_c_end = pt_temp;
      end
      n_search = round(norm(pt_c_end - pt_c_start));
      
      % 매칭쌍을 찾는다
      scores       = -ones(1,n_search); % 서칭하는 곳의 모든 score저장.
      score_best   = -1e5;
      pt_c_over = []; % thres2를 (0.8)을 넘은 점들을 모아둠.
      pt_c_best = zeros(2,1); % 매칭된 점의 위치. (subpixel)
      
      % 범위를 순회하며 서칭.
      ind_best = -1;
      for j = 1:n_search
         % 현재 좌표
         pt_c_now =  j*l_c + pt_c_start;
         % 패치를 구한다.
         pts_c_patch = calcEpiPatchTemplate(pt_c_now, l_c, pts_patch_template);
         patch_c     = interpImage(img_c, pts_c_patch);         
         % 스코어를 구한다.
         scores(j)   = calcZNCC(patch_k(:), patch_c(:));
         % 스코어를 비교한다.
         if(scores(j) > thres_zncc2)
            pt_c_over = [pt_c_over, pt_c_now];
         end
         % 만약, 현재 ZNCC가 이전 값 보다 높으면 바꾼다.
         if(scores(j) > score_best)
            score_best = scores(j);
            pt_c_best = pt_c_now;
            ind_best = j;
         end
      end
      
      % 최고 점수가 threshold를 넘고, 모든 u_over가 6 pixel 이내이고,
      % 범위 가장자리가 아닌 경우에만 신뢰.
      if(n_search < 4)
         flag_valid(i)=0;
      end
      if(ind_best < 2 || ind_best > n_search-2)
         flag_valid(i)=0;
      end
      if(flag_valid(i))
         for j = 1:size(pt_c_over,2) % 하나라도 6 px 이상 벗어나있다면 기각.
            if(norm(pt_c_best - pt_c_over(:,j)) > thres_multipeak)
               flag_valid(i) = 0;
               break;
            end
         end
      end
      
      if(flag_valid(i) && (score_best > thres_zncc1))
         % 최고 점수를 저장한다.
         scores_best(i) = score_best;
         % score history 이용해서 2차함수 모델로 subpixel refinement 수행.
         s1 = scores(ind_best-1); % 왼쪽 값.
         s2 = scores(ind_best); % 중심 값.
         s3 = scores(ind_best+1); % 우측 값.
         pt_c_best = pt_c_best - (s3-s1)/(s3+s1-2*s2)*0.5*l_c; % refine된 좌표 값.
         
         X = triangulatePointsDLT(pt_k,pt_c_best,g_ck);
         
         % inverse depth를 계산한다.
         invd_save(i) = 1/X(3);
         
         % standard deviation을 계산한다.
         std_invd_save(i) = 1/abs(cos_th)*sqrt(eps_edge^2 + eps_epi^2*(1-cos_th^2))*bfinv;
      else
         flag_valid(i) = 0;
      end
   end
end

% % 결과 업데이트
for k = 1:n_pts
   if(flag_valid(k) > 0)
      invd_prev = frame_k.left.invd(k);
      std_prev = frame_k.left.std_invd(k);

      invd_curr = invd_save(k);
      std_curr  = std_invd_save(k);
      if(invd_prev > 0) % (1) inverse depth가 있던 경우.
         % 퓨전.
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

% 결과 업데이트
% for k = 1:n_pts
%    if(flag_valid(k) > 0)
%       invd_prev = frame_k.left.invd(k);
%       std_prev = frame_k.left.std_invd(k);
%       
%       invd_curr = invd_save(k);
%       std_curr  = std_invd_save(k);
%       if(invd_prev > 0) % (1) inverse depth가 있던 경우.
%          % 퓨전.
%          a_prev   = frame_k.left.df.a(k);
%          b_prev   = frame_k.left.df.b(k);
%          mu_prev  = frame_k.left.df.mu(k);
%          sig_prev = frame_k.left.df.sig(k);
%          z_min_prev = frame_k.left.df.z_min(k);
%          z_max_prev = frame_k.left.df.z_max(k);
%          [a_new, b_new, mu_new, sig_new, z_min_new, z_max_new]...
%             =updateDF(invd_curr, std_curr,a_prev,b_prev, mu_prev, sig_prev, z_min_prev, z_max_prev);
%          frame_k.left.df.a(k) = a_new;
%          frame_k.left.df.b(k) = b_new;
%          frame_k.left.df.mu(k) = mu_new;
%          frame_k.left.df.sig(k) = sig_new;
%          frame_k.left.df.z_min(k) = z_min_new;
%          frame_k.left.df.z_max(k) = z_max_new;
%          
%          %          invd_fusion = 1/(std_prev^2+std_curr^2)*(invd_prev*std_curr^2 + invd_curr*std_prev^2);
%          %          std_fusion   = sqrt( (std_prev^2*std_curr^2)/(std_prev^2+std_curr^2) );
%          
%          % inverse depth를 계산한다.
%          frame_k.left.invd(k)     = mu_new;
%          % standard deviation을 계산한다.
%          frame_k.left.std_invd(k) = sig_new;
%          % pts3d를 계산한다.
%          frame_k.left.pts3d(:,k) = Kinv*[frame_k.left.pts(:,k);1]/mu_new;
%          frame_k.left.is_recon(k) = 1;
%          
%       else % 아무것도 없는 경우.
%          frame_k.left.df.mu(k) = invd_curr;
%          frame_k.left.df.sig(k) = std_curr;
%          if(frame_k.left.df.z_min(k) > invd_curr)
%             frame_k.left.df.z_min(k) = invd_curr;
%          end
%          if(frame_k.left.df.z_max(k) < invd_curr)
%             frame_k.left.df.z_max(k) = invd_curr;
%          end
%          % inverse depth를 계산한다.
%          frame_k.left.invd(k)     = invd_curr;
%          % standard deviation을 계산한다.
%          frame_k.left.std_invd(k) = std_curr;
%          % pts3d를 계산한다.
%          frame_k.left.pts3d(:,k)  = Kinv*[frame_k.left.pts(:,k);1]/invd_curr;
%          frame_k.left.is_recon(k) = 1;
%       end
%    end
% end
end