% Specify the folder containing images
folder = '../../IQA/data/LIVEitW/images';

% Get a list of all image files in the folder
imageFiles = dir(fullfile(folder, '*.jpg')); % Modify file extension if needed

% Step 4: Iterate through each image file and calculate BRISQUE score
image_names = strings(numel(imageFiles), 1); % Initialize cell array to store image names
brisque_scores = zeros(numel(imageFiles), 1); % Initialize array to store BRISQUE scores

for i = 1:numel(imageFiles)
    % Read the image
    img = imread(fullfile(folder, imageFiles(i).name));
    
    % Calculate BRISQUE score
    brisque_score = brisque(img); % [0,100], 0-better
    brisque_score = 5 - brisque_score / 25; % [1,5], 1-worse

    % Display BRISQUE score
    % fprintf('Image: %s, BRISQUE score: %.4f\n', imageFiles(i).name, brisque_score);
    
    % Store BRISQUE score along with filename
    image_names(i) = imageFiles(i).name;
    brisque_scores(i) = brisque_score;
end
