% This script simulates speckle patterns by convolving an object image 
% with an experimentally captured Point Spread Function (PSF).
% The object is resized to an equivalent pixel size based on system parameters.

clc; clear; close all;

%% --- 1. Configuration of Experimental Parameters ---
% Parameters should match your actual experimental setup
dist_u = 230;          % Object distance (u) in mm
dist_v = 15;          % Image distance (v) in mm
pixel_lcd = 116.55;       % lcd pixel size in micrometers
pixel_cam = 2.90;      % Camera pixel size in micrometers
lcd_obj_res = 32;         % Object resolution loaded on LCD (26,32,64,80)
target_res = 384;      % Final output resolution for Deep Learning model
input_image_path = 'lb_1.png'; % Path to your ground truth image
psf_path = 'PSF.png';     % Path to your experimental PSF
realsp_path = 'preprocessed_real_speckle.png';     % Path to your experimental speckle
%% --- 2. Calculate Equivalent Scaling Factor ---
% Magnification M = v / u
magnification = dist_v / dist_u;

% Scaling factor: How many camera pixels represent one lcd pixel
% ratio = (Physical size of lcd pixel at camera plane) / (Camera pixel size)
scaling_ratio = (pixel_lcd * magnification) / pixel_cam;
equivalent_obj_size = round(lcd_obj_res * scaling_ratio);

fprintf('System Magnification: %.2f\n', magnification);
fprintf('Equivalent Scaling Ratio: %.4f\n', scaling_ratio);

%% --- 3. Image Pre-processing ---
obj = im2double(imread(input_image_path));
if size(obj, 3) > 1, obj = rgb2gray(obj); end

psf = im2double(imread(psf_path));
if size(psf, 3) > 1, psf = rgb2gray(psf); end
% psf_cropped = center_crop(psf, 1024, 1024);

realsp = im2double(imread(realsp_path));
if size(realsp, 3) > 1, realsp = rgb2gray(realsp); end

% Step 1: Downsample the ground truth to match LCD's 32x32 physical grid
obj_lcd = imresize(obj, [lcd_obj_res, lcd_obj_res], 'bicubic');

% Step 2: Scale the 32x32 object to its equivalent size in camera domain
obj_camera_domain = imresize(obj_lcd, [equivalent_obj_size, equivalent_obj_size], 'bicubic');

%% --- 4. Speckle Simulation (Convolution) ---
speckle_full = conv2(obj_camera_domain, psf, 'full');

%% --- 5. Post-processing ---
speckle_cropped = center_crop(speckle_full, target_res, target_res);
speckle_final = maxnorm(speckle_cropped);

%% --- 6. Visualization ---
figure('Name', 'Speckle Generation Pipeline');
subplot(2,2,1); imshow(obj, []); title('Original Object');
subplot(2,2,2); imshow(psf, []); title('PSF');
subplot(2,2,3); imshow(speckle_final, []); title('Final 384x384 Speckle');
subplot(2,2,4); imshow(realsp, []); title('Preprocessed Real Speckle');
% Save the generated data
% imwrite(im2uint8(speckle_final), 'generated_speckle.png');

%% --- Functions ---
function cropped_image = center_crop(image, target_height, target_width, x_shift, y_shift)

    if nargin < 5
        y_shift = 0; 
    end
    if nargin < 4
        x_shift = 0; 
    end

    if ~ismatrix(image) && ~ndims(image) == 3
        error('rgb error');
    end

    [height, width, ~] = size(image);

    if target_height > height || target_width > width
        error('size error');
    end

    start_x = floor((width - target_width) / 2) + 1 + x_shift;
    start_y = floor((height - target_height) / 2) + 1 + y_shift;

    end_x = start_x + target_width - 1;
    end_y = start_y + target_height - 1;

    if start_x < 1 || start_y < 1 || end_x > width || end_y > height
        error('boundray error');
    end

    cropped_image = image(start_y:end_y, start_x:end_x, :);
end

function out = maxnorm(in)
    out = (in - min(in(:))) / (max(in(:)) - min(in(:)) + eps);
end