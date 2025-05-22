function [score] = niqe(root_path, image)
    
    niqe_path = strcat(root_path, '+niqe\\');
    load(strcat(niqe_path, 'modelparameters.mat'));
 
    blocksizerow    = 96;
    blocksizecol    = 96;
    blockrowoverlap = 0;
    blockcoloverlap = 0;

    score = niqe.computequality(image,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    clc
end