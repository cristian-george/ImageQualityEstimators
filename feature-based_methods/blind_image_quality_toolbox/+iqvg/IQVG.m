function [score] = IQVG(root_path, img)

    import iqvg.*;
    
    svm_scale_path = strcat(root_path, 'libsvm\\3.22\\windows\\svm-scale.exe');
    iqvg_path = strcat(root_path, '+iqvg\\');

    scale_file = strcat(iqvg_path, 'scale_parameter');
    input_file = strcat(iqvg_path, 'f.txt');
    scaled_file = strcat(iqvg_path, 'f_scaled.txt');

	patchSize = 7;
	maxFrequency = 1;
	frequencyNum = 5;
	orientionsNum = 4; 
	patchNum = 5000; % more patches, more stable the result is
	gfFeature = sample_img(img, patchSize, maxFrequency, frequencyNum, orientionsNum, patchNum);
	% 200 is the number of bins that we use to encode the Gabor feature.
	imgPre = buildHistogram(gfFeature, patchNum, 200);
	outSVMData(input_file, imgPre);
    
    scaleCommand = sprintf('"%s" -r "%s" "%s" > "%s"', svm_scale_path, scale_file, input_file, scaled_file);
    system(scaleCommand);
    
	[~, f_scaled] = libsvmread(scaled_file);

	load(strcat(iqvg_path, 'model.mat'), 'model');
	[predict_label, ~, ~] = svmpredict(-1, f_scaled, model, '-b 0');
	score = predict_label;
    
    delete(input_file);
    delete(scaled_file);
    clc
