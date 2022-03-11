%% Read input cloud
cloud_A = '/tmp/sph_image.pcd';
cloud_B = '/tmp/sph_cloud.pcd';

A = pcread(cloud_A);
B = pcread(cloud_B);

%% Show them indiviually

clf;
subplot(1,2,1);
pcshow(A);
subplot(1,2,2);
pcshow(B);


%% Overlab both modalities
mask_a = any(A.Color, 2);
mask_b = any(B.Color, 2);
rgb = [0.2126; 0.7152; 0.0722];

xyz_a = A.Location(mask_a, :);
xyz_b = B.Location(mask_b, :);
% i_a = single(A.Color(mask_a, :)) * rgb;
c_a = A.Color(mask_a, :);
i_b = single(B.Color(mask_b, :)) * rgb;

% AA = pointCloud(xyz_a, 'Intensity', i_a);
AA = pointCloud(xyz_a, 'Color', c_a);
BB = pointCloud(xyz_b, 'Intensity', i_b);

% pcshowpair(AA, BB);
pcshow(AA)
hold on;
pcshow(BB);
hold off;

%% Overlap both modalities with transparency
mask_a = any(A.Color, 2);
mask_b = any(B.Color, 2);
rgb = [0.2126; 0.7152; 0.0722];

xyz_a = A.Location(mask_a, :);
xyz_b = B.Location(mask_b, :);
% i_a = single(A.Color(mask_a, :)) * rgb;
c_a = A.Color(mask_a, :);
i_b = single(B.Color(mask_b, :)) * rgb;

% AA = pointCloud(xyz_a, 'Intensity', i_a);
AA = pointCloud(xyz_a, 'Color', c_a);
BB = pointCloud(xyz_b, 'Intensity', i_b);

pcshow(AA)
hold on;
sc = scatter3(BB.Location(:,1), BB.Location(:,2), BB.Location(:,3),[], i_b);
sc.MarkerFaceAlpha = .1;
sc.MarkerEdgeAlpha = .1;

hold off;