clear;clc;close all;
%% Preprocessing Scheme
% Configuration
input_path = 'speckle.png'; % Replace with your image path
output_path = 'processed_speckle.png';
conv_path = "convspeckle.png";
block_size = 128;         % Block size for background estimation
corr_factor = 0.3;       % Correction coefficient (c)
bottleneck_res = 48;     % Resolution for interpolation-based denoising
target_res = 384;        % Final output

%% 1. Load Image
sp = im2double(imread(input_path));
conv = im2double(imread(conv_path));
if size(sp, 3) > 1, sp = rgb2gray(sp); end
if size(conv, 3) > 1, conv = rgb2gray(conv); end

%% 2. Background Correction (Grayscale Correction)
sp_corrected = correctBackground(sp, block_size, corr_factor);
conv_corrected = correctBackground(conv, block_size, corr_factor);

%% 3. Interpolation-based Denoising (384 -> 64 -> 384)
sp_low = imresize(sp_corrected, [bottleneck_res, bottleneck_res], 'bilinear');
sp_denoised = imresize(sp_low, [target_res, target_res], 'bicubic');

%% 4. Final Normalization
sp_final = maxnorm(sp_denoised);
conv_final = maxnorm(conv_corrected);

%% 5. Visualization
figure;
subplot(2,2,1); imshow(sp, []); title('Original Speckle');
subplot(2,2,2); imshow(sp_corrected, []); title('Background Corrected');
subplot(2,2,3); imshow(sp_final, []); title('Final Denoised');
subplot(2,2,4); imshow(conv_corrected, []); title('Conv Speckle');

% Save result
% imwrite(im2uint8(img_final), output_path);

%% --- Functions ---

function I_out = correctBackground(im, blocksize, c)
    % Estimates local background using block-based intensity statistics
    [m, n] = size(im);
    blocknum1 = floor(m/blocksize);
    blocknum2 = floor(n/blocksize);
    
    % Crop image to fit block multiples
    L = blocknum1 * blocksize;
    H = blocknum2 * blocksize;
    im_cropped = im(1:L, 1:H);
    
    % Background estimation matrix
    bg_map = zeros(blocknum1, blocknum2);
    
    for k = 1:blocknum2
        for h = 1:blocknum1
            % Extract block
            row_idx = (1 + blocksize*(h-1)) : (blocksize*h);
            col_idx = (1 + blocksize*(k-1)) : (blocksize*k);
            block = im_cropped(row_idx, col_idx);
            
            % Use top 75% intensity values to estimate background brightness
            % This avoids underestimation caused by dark speckle regions
            block_sorted = sort(block(:), 'ascend');
            threshold_idx = floor(numel(block_sorted) / 4);
            bg_map(h,k) = mean(block_sorted(threshold_idx:end));
        end
    end
    
    % Resize background map to original dimensions using bilinear interpolation
    bg_full = imresize(bg_map, [L, H], 'bilinear');
    
    % Apply correction formula: I_out = I_in * c / I_bg
    I_out = (im_cropped .* c) ./ (bg_full + eps); % Add eps to avoid div by zero
end

function A_n = maxnorm(A)
    % Min-Max normalization to [0, 1]
    minA = min(A(:));
    maxA = max(A(:));
    A_n = (A - minA) / (maxA - minA + eps);
end
