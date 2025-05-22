
root_path = 'D:\\Files\\Projects\\Image Quality Assessment\\IQA\\';

% Path to toolbox metrics
metrics_path = strcat(root_path, 'IQM\\blind_image_quality_toolbox\\');
% Path to IQA datasets
datasets_path = strcat(root_path, 'IQA\\data\\datasets\\');

% Modify these two variables to compute quality metrics for your dataset
dataset = 'Prisma';
test = 'images';

folder_path = fullfile(strcat(datasets_path, dataset), test);

% Define the file extensions you want to include
extensions = {'*.jpg', '*.png'};

% Loop over each extension and collect the file paths
image_paths = {};
for i = 1:length(extensions)
    % Get the list of files for the current extension
    image_files = dir(fullfile(folder_path, extensions{i}));
    
    % Get the full paths and append them to the image_paths array
    for j = 1:numel(image_files)
        image_paths{end+1} = fullfile(folder_path, image_files(j).name);
    end
end

% Initialize cell array to store the output table rows
output_data = {};

for i = 1:numel(image_paths)
    img = imread(image_paths{i});
    metrics = computeQualityMetrics(metrics_path, img);
    
    % Extract the image name
    [~, image_name, ext] = fileparts(image_paths{i});
    
     % Convert biqaa_rgb vector to a comma-separated string
    % biqaa_rgb_str = sprintf('%f,%f,%f', metrics.biqaa_rgb);

    % Append the image name and metrics to the output_data cell array
    output_data{end+1, 1} = [image_name, ext];
    output_data{end, 2} = metrics.biqaa_gray;
    output_data{end, 3} = metrics.biqaa_rgb; %biqaa_rgb_str;
    output_data{end, 4} = metrics.biqi;
    output_data{end, 5} = metrics.bliinds;
    output_data{end, 6} = metrics.brisque;
    output_data{end, 7} = metrics.divine;
    output_data{end, 8} = metrics.iqvg;
    output_data{end, 9} = metrics.niqe;
end

column_names = {'image_name', 'biqaa_gray', 'biqaa_rgb', 'biqi', 'bliinds', 'brisque', 'divine', 'iqvg', 'niqe'};
results_table = cell2table(output_data, 'VariableNames', column_names);

csv_filename = fullfile(strcat(datasets_path, dataset), 'image_quality_metrics.csv');
writetable(results_table, csv_filename);
