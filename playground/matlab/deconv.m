%% Read and transform input image

i = im2double(imread('cameraman.tif'));
rows = size(i, 1);
columns = size(i, 2);
h = fspecial('disk',1);

% PSF = fspecial('gaussian',5,5);
filter = fspecial('gaussian', [rows, columns], 0.1*rows);
% filter = fftshift(fft2(h,rows,columns));

%% Filter
I = fftshift(fft2(i));
I_hat = I .* filter;
i_hat = real(ifft2(ifftshift(I_hat)));

%% Add noise to img

noise_db = 30;
sigma_u = 10^(-noise_db/20)*abs(1-0);
noise = sigma_u*randn(size(i_hat));
i_hat = i_hat + noise;
I_hat = fftshift(fft2(i_hat));

%% Visualize 
subplot(2, 2, 1);
imagesc(i);
colorbar;
subplot(2, 2, 2)
imagesc(real(filter));
colorbar;
subplot(2, 2, 3)
imagesc(i_hat);
colorbar;
subplot(2, 2, 4)
% imagesc(real(ifft2(ifftshift(filter))));
imagesc(real(fft2(i)));
colorbar;

%% Wiener Deconvolution
I_recon = I_hat ./ filter;
i_recon = real(ifft2(ifftshift(I_recon)));

% subplot(2, 2, 1);
% imagesc(i);
% colorbar;
subplot(2, 2, 4)
imagesc(i_recon);
colorbar;

%% Wiener Deconvolution

numerator = conj(filter).*I_hat;
denominator = conj(filter).*filter + 1e-2;

i_recon = real(ifft2(ifftshift(numerator./denominator)));

% subplot(2, 2, 1);
% imagesc(i);
% colorbar;
subplot(2, 2, 4)
imagesc(i_recon);
colorbar;

%% SVD Wiener

% [U,D,V] = svds(filter, 10);
[U,D,V] = svd(filter);
D_inv = pinv(D);

rows = size(D,1);
cols = size(D,2);
rows = 15;
cols = 15;
for i = 1:rows
  for j = 1:cols
    D(i,j) = 0;
  end
end

I_recon = I_hat ./ (V * D * U');
% I_recon = D_inv * (V * U') .* I_hat;
% I_recon = pinv(filter).*I_hat;
i_recon = real(ifft2(ifftshift(I_recon)));

% subplot(2, 2, 1);
% imagesc(i);
% colorbar;
subplot(2, 2, 4)
imagesc(i_recon);
colorbar;

