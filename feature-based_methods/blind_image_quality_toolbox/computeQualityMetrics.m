function [results] = computeQualityMetrics(root_path, image)
    
    % Cjlin Libsvm v3.22 at https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/libsvm-3.22.zip
    cjlin1_libsvm_path = strcat(root_path,'libsvm\\3.22\\matlab');
    % Gregreeman Libsvm at https://github.com/gregfreeman/libsvm/tree/new_matlab_interface
    gregfreeman_libsvm_path = strcat(root_path,'libsvm\\3.12\\matlab');


    % Anisotrophy Test
    [gray,rgb] = biqaa.blindimagequality(image,8,6,0,'degree');
    results.biqaa_gray = gray;
    results.biqaa_rgb = rgb;

    % BIQI Test
    results.biqi = biqi.biqi(root_path, image);

    % Bliinds2 Test
    results.bliinds = bliinds2.bliinds2_score(image);

    % BRISQUE Test
    addpath(cjlin1_libsvm_path);
    results.brisque = brisque.brisquescore(root_path, image);
    rmpath(cjlin1_libsvm_path);

    % DIVINE Test
    addpath(gregfreeman_libsvm_path);
    results.divine = divine.divine(root_path, image);
    rmpath(gregfreeman_libsvm_path);

    % IQVG Test
    addpath(cjlin1_libsvm_path);
    results.iqvg = iqvg.IQVG(root_path, image);
    rmpath(cjlin1_libsvm_path);

    % NIQE Test
    results.niqe = niqe.niqe(root_path, image);
end
