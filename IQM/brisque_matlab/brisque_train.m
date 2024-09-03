% Specify the path to the directory containing images
image_dir = '../../IQA/data/KonIQ-10K/train';

% Specify the path to your CSV file containing image names and scores
csv_file = '../../IQA/data/KonIQ-10K/train_labels.csv';

% Read the CSV file into a table
data_table = readtable(csv_file);

% Access the columns 'image_name' and 'score'
image_names = data_table.image_name;
image_scores = data_table.MOS;
scores = 25 * (5 - image_scores);

% Create an ImageDatastore object for the images
imds = imageDatastore(fullfile(image_dir, image_names));

% Train a BRISQUE model
model = fitbrisque(imds, scores);

%% 

% Specify the folder containing images
test_folder = '../../IQA/data/LIVEitW/images';

% Get a list of all image files in the folder
imageFiles = dir(fullfile(test_folder, '*.jpg')); % Modify file extension if needed

% Step 4: Iterate through each image file and calculate BRISQUE score
image_names = strings(numel(imageFiles), 1); % Initialize cell array to store image names
brisque_scores = zeros(numel(imageFiles), 1); % Initialize array to store BRISQUE scores

for i = 1:numel(imageFiles)
    % Read the image
    img = imread(fullfile(test_folder, imageFiles(i).name));
    
    % Calculate BRISQUE score
    brisque_score = brisque(img, model); % [0,100], 0-better
    brisque_score = 5 - brisque_score / 25; % [1,5], 1-worse

    % Display BRISQUE score
    fprintf('Image: %s, BRISQUE score: %.4f\n', imageFiles(i).name, brisque_score);
    
    % Store BRISQUE score along with filename
    image_names(i) = imageFiles(i).name;
    brisque_scores(i) = brisque_score;
end
