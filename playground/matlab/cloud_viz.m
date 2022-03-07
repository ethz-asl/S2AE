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

%% Show them combined

clf;
ax = pcshow(A);
hold on;

%%
mask_a = any(A.Color, 2);
mask_b = any(B.Color, 2);
rgb = [0.2126; 0.7152; 0.0722];

xyz_a = A.Location(mask_a, :);
xyz_b = B.Location(mask_b, :);
i_a = single(A.Color(mask_a, :)) * rgb;
c_a = A.Color(mask_a, :);
i_b = single(B.Color(mask_b, :)) * rgb;

% AA = pointCloud(xyz_a, 'Intensity', i_a);
AA = pointCloud(xyz_a, 'Color', c_a);
BB = pointCloud(xyz_b, 'Intensity', i_b);

% pcshowpair(AA, BB);
pcshow(AA)