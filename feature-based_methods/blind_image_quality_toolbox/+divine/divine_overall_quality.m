function q = divine_overall_quality(divine_path, r)

    % Function to compute overall quality given feature vector 'r'
    
    load(strcat(divine_path,'data_live_trained.mat'))
    
    %% Classification
    atrain = repmat(a_class,[size(r,1) 1]);btrain = repmat(b_class,[size(r,1) 1]);
    x_curr = atrain.*r+btrain;
    
    % [pred_class acc p] = svmpredict(1,x_curr,model_class,'-b 1');
    % use `git clone git://github.com/gregfreeman/libsvm.git -b new_matlab_interface`
    
    [~, p] = svmpredict(x_curr,model_class,struct('output','probability'));
    
    
    %% Regression
    for i = 1:5
        atrain = repmat(a_reg(i,:),[size(r,1) 1]);btrain = repmat(b_reg(i,:),[size(r,1) 1]);
        x_curr = atrain.*r+btrain;
    %    [q(i) reg_acc(i,:)] = svmpredict(1,x_curr,model_reg{i});
        [q(i)] = svmpredict(x_curr,model_reg{i});
    end
    %% Final Score
    q = sum(p.*q);
    clc
