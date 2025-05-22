function brisque_score  = brisquescore(root_path, imdist)

    import brisque.*;
    
    % Paths to libsvm executables (svmscale and svmpredict)
    svm_scale_path = strcat(root_path, 'libsvm\\3.22\\windows\\svm-scale.exe');
    svm_predict_path = strcat(root_path, 'libsvm\\3.22\\windows\\svm-predict.exe');
    
    % Paths to input and output files
    brisque_path = strcat(root_path, '+brisque\\');
    input_file = strcat(brisque_path, 'test_ind');
    scaled_file = strcat(brisque_path, 'test_ind_scaled');
    output_file = strcat(brisque_path, 'output');
    
    
    if(size(imdist,3)==3)
        imdist = uint8(imdist);
        imdist = rgb2gray(imdist);
    end
    
    imdist = double(imdist);
    feat = brisque_feature(imdist);
    disp('feat computed')
    
    
    %---------------------------------------------------------------------
    % Quality Score Computation
    %---------------------------------------------------------------------
    
    
    fid = fopen(input_file,'w');
    
    for jj = 1:size(feat,1)
        
    fprintf(fid,'1 ');
    for kk = 1:size(feat,2)
    fprintf(fid,'%d:%f ',kk,feat(jj,kk));
    end
    fprintf(fid,'\n');
    end
    
    fclose(fid);
    warning off all
    
    range_file = strcat(brisque_path, 'allrange');
    model_file = strcat(brisque_path, 'allmodel');
    
    scale_command = sprintf('"%s" -r "%s" "%s" >> "%s"', svm_scale_path, range_file, input_file, scaled_file);
    predict_command = sprintf('"%s" -b 1 "%s" "%s" "%s"', svm_predict_path, scaled_file, model_file, output_file);
    system(scale_command);
    system(predict_command);
    
    load(output_file, 'output')
    brisque_score = output;
    
    delete(input_file)
    delete(scaled_file)
    delete(output_file)
    clc
end
