function [quality, probs] = biqi(root_path, im)

%========================================================================
%
% -----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
% Copyright (c) 2009 The University of Texas at Austin
% All rights reserved.
% 
% Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
% modify, and distribute this code (the source files) and its documentation for
% any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
% original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
% and Center for Perceptual Systems (CPS, http://www.cps.utexas.edu) at the University of Texas at Austin (UT Austin, 
% http://www.utexas.edu), is acknowledged in any publication that reports research using this code. The research
% is to be cited in the bibliography as:
% 
% 1. A. K. Moorthy and A. C. Bovik, "A Modular Framework for Constructing Blind
% Universal Quality Indices", submitted to IEEE Signal Processing Letters (2009).
% 
% 2. A. K. Moorthy and A. C. Bovik, "BIQI Software Release", 
% URL: http://live.ece.utexas.edu/research/quality/biqi.zip, 2009.
% 
% IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
% OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
% AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
% AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
% 
% -----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

%Author  : Anush Krishna Moorthy
%Version : 1.0
% 
%The authors are with the Laboratory for Image and Video Engineering
%(LIVE), Department of Electrical and Computer Engineering, The
%University of Texas at Austin, Austin, TX.
%
%Kindly report any suggestions or corrections to anushmoorthy@gmail.com
%
%========================================================================
%
% This is a demonstration of the Blind Image Quality Index (BIQI) . 
% It is an implementation of the BIQI in the reference.
% The algorithm is described in:
% A. K. Moorthy and A. C. Bovik, "A Modular Framework for Constructing Blind
% Universal Quality Indices", submitted to IEEE Signal Processing Letters (2009).
%
%You can change this program as you like and use it anywhere, but please
%refer to its original source (cite our paper and our web page at
% http://live.ece.utexas.edu/research/quality/biqi.zip).
%
%Input : A test 8bits/pixel grayscale image loaded in a 2-D array
%Output: A quality score of the image. The score typically has a value
%        between 0 and 100 (0 represents the best quality, 100 the worst).
%
%Usage:
%
%1. Load the image, for example
%
%   image = rgb2gray(imread('testimage.jpg')); 
%
%2. Call this function to calculate the quality score:
%
%    quality = biqi(image)
%
% Dependencies:
% MATLAB Wavelet  Toolbox
% You may need the MATLAB Image Processing Toolbox
% Binaries: svm-train, svm-scale (from LibSVM) - provided with release
% Other m files: jpeg_quality_score.m (provided with release)
% Data files: range2, range2_wn, range2_blur, range2_jp2k, model_89,
% model_89_wn, model_89_blur, model_89_jp2k, rang2_ff model_ff
%========================================================================


    import biqi.*
    
    % Paths to libsvm executables (svmscale and svmpredict)
    svm_scale_path = strcat(root_path, 'libsvm\\3.22\\windows\\svm-scale.exe');
    svm_predict_path = strcat(root_path, 'libsvm\\3.22\\windows\\svm-predict.exe');
    
    % Paths to input and output files
    biqi_path = strcat(root_path, '+biqi\\');
    input_file = strcat(biqi_path, 'test_ind.txt');
    scaled_file = strcat(biqi_path, 'test_ind_scaled');
    output_file = strcat(biqi_path, 'output_blur');
    output_89_file = strcat(biqi_path, 'output_89');
    
    
    if(size(im,3)~=1)
        im = rgb2gray(im);
    end
    
    %% First compute statistics
    num_scales = 3; % somethings are hardcoded for this...please be careful when changing.
    gam = 0.2:0.001:10;
    r_gam = gamma(1./gam).*gamma(3./gam)./(gamma(2./gam)).^2;
    
    [C, S] = wavedec2(im,num_scales,'db9');
    for p = 1:num_scales
        [horz_temp,vert_temp,diag_temp] = detcoef2('all',C,S,p) ;
        horz(p) = {[horz_temp(:)]};
        diag(p) = {[diag_temp(:)]};
        vert(p) = {[vert_temp(:)]};
    
        h_horz_curr  = cell2mat(horz(p));
        h_vert_curr  = cell2mat(vert(p));
        h_diag_curr  = cell2mat(diag(p));
    
        mu_horz(p) = mean(h_horz_curr);
        sigma_sq_horz(p)  = mean((h_horz_curr-mu_horz(p)).^2);
        E_horz = mean(abs(h_horz_curr-mu_horz(p)));
        rho_horz = sigma_sq_horz(p)/E_horz^2;
        [~, array_position] = min(abs(rho_horz - r_gam));
        gam_horz(p) = gam(array_position);
    
        mu_vert(p) = mean(h_vert_curr);
        sigma_sq_vert(p)  = mean((h_vert_curr-mu_vert(p)).^2);
        E_vert = mean(abs(h_vert_curr-mu_vert(p)));
        rho_vert = sigma_sq_vert(p)/E_vert^2;
        [~, array_position] = min(abs(rho_vert - r_gam));
        gam_vert(p) = gam(array_position);
    
        mu_diag(p) = mean(h_diag_curr);
        sigma_sq_diag(p)  = mean((h_diag_curr-mu_diag(p)).^2);
        E_diag = mean(abs(h_diag_curr-mu_diag(p)));
        rho_diag = sigma_sq_diag(p)/E_diag^2;
        [~, array_position] = min(abs(rho_diag - r_gam));
        gam_diag(p) = gam(array_position);
    end
    rep_vec = [mu_horz mu_vert mu_diag sigma_sq_horz sigma_sq_vert sigma_sq_diag gam_horz gam_vert gam_diag];
    rep_vec(:,1:9) = []; % remove the means...
    %% Now classify
    
    fid = fopen(input_file,'w');
    for j = 1:size(rep_vec,1)
        fprintf(fid,'%d ',j);
        for k = 1:size(rep_vec,2)
            fprintf(fid,'%d:%f ',k,rep_vec(j,k));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    
    range_file = strcat(biqi_path, 'range2');
    model_file = strcat(biqi_path, 'model_89');
    
    scale_command = sprintf('"%s" -r "%s" "%s" >> "%s"', svm_scale_path, range_file, input_file, scaled_file);
    predict_command = sprintf('"%s" -b 1 "%s" "%s" "%s"', svm_predict_path, scaled_file, model_file, output_89_file);
    system(scale_command);
    system(predict_command);
    
    delete(input_file) 
    delete(scaled_file)
    
    %% Quality along each dimension
    
    % Write out SVM compatible
    
    fid = fopen(input_file,'w');
    for j = 1:size(rep_vec,1)
        fprintf(fid,'%f ',j);
        for k = 1:size(rep_vec,2)
            fprintf(fid,'%d:%f ',k,rep_vec(j,k));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    
    % ----Jp2k quality ----
    range_file = strcat(biqi_path, 'range2_jp2k');
    model_file = strcat(biqi_path, 'model_89_jp2k');
    
    scale_command = sprintf('"%s" -r "%s" "%s" >> "%s"', svm_scale_path, range_file, input_file, scaled_file);
    predict_command = sprintf('"%s" -b 1 "%s" "%s" "%s"', svm_predict_path, scaled_file, model_file, output_file);
    system(scale_command);
    system(predict_command);
    
    load(output_file, 'output_blur')
    jp2k_score = output_blur;
    
    delete(output_file) 
    delete(scaled_file)
    
    
    % ---- JPEG quality ----
    jpeg_score = jpeg_quality_score(im);
    
    
    % ---- WN quality ----
    range_file = strcat(biqi_path, 'range2_wn');
    model_file = strcat(biqi_path, 'model_89_wn');
    
    scale_command = sprintf('"%s" -r "%s" "%s" >> "%s"', svm_scale_path, range_file, input_file, scaled_file);
    predict_command = sprintf('"%s" -b 1 "%s" "%s" "%s"', svm_predict_path, scaled_file, model_file, output_file);
    system(scale_command);
    system(predict_command);
    
    load(output_file, 'output_blur')
    wn_score = output_blur;
    
    delete(output_file) 
    delete(scaled_file)
    
    
    % ---- Blur quality ----
    range_file = strcat(biqi_path, 'range2_blur');
    model_file = strcat(biqi_path, 'model_89_blur');
    
    scale_command = sprintf('"%s" -r "%s" "%s" >> "%s"', svm_scale_path, range_file, input_file, scaled_file);
    predict_command = sprintf('"%s" -b 1 "%s" "%s" "%s"', svm_predict_path, scaled_file, model_file, output_file);
    system(scale_command);
    system(predict_command);
    
    load(output_file, 'output_blur')
    blur_score = output_blur;
    
    delete(output_file) 
    delete(scaled_file)
    
    
    % ---- FF quality ----
    range_file = strcat(biqi_path, 'range2_ff');
    model_file = strcat(biqi_path, 'model_89_ff');
    
    scale_command = sprintf('"%s" -r "%s" "%s" >> "%s"', svm_scale_path, range_file, input_file, scaled_file);
    predict_command = sprintf('"%s" -b 1 "%s" "%s" "%s"', svm_predict_path, scaled_file, model_file, output_file);
    system(scale_command);
    system(predict_command);
    
    load(output_file, 'output_blur')
    ff_score = output_blur;
    
    delete(input_file)
    delete(output_file) 
    delete(scaled_file)
    
    %% Final pooling
    
    % figure out probabilities
    fid = fopen(output_89_file,'r');
    fgetl(fid);
    C = textscan(fid,'%f %f %f %f %f %f');
    output = [C{1} C{2} C{3} C{4} C{5} C{6}];
    fclose(fid);
    probs = output(:,2:end);
    scores  = [jp2k_score, jpeg_score, wn_score, blur_score, ff_score];
    quality = sum(probs.*scores,2);
    delete(output_89_file)
    clc
end
