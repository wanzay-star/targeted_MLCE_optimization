

%% Channel Estimation for LMS, Linear Regression(LG) and zoh methods
% It uses differenced samples & sweeps across latency values

clear 

%% Load Datasets for each turbulence regime.

addpath('.\data')

% load('lin_wan5_strong_turb_samps.mat'); 
% dataset = 'Strong Turbulence';
% X = lin_wan5_s_dat(1:1e6); 

load('lin_wan5_mod_turb_samps.mat'); 
dataset = 'Moderate Turbulence';
X = lin_wan5_m_dat; 

% load('lin_wan5_weak_turb_samps.mat'); 
% dataset = 'Weak Turbulence';
% X = lin_wan5_w_dat(1:1e6); 

%%

Fs_meas = 1e4;  % Sample frequency
Fs = Fs_meas/1;
X = X(1:Fs_meas/Fs:end);

wa = pow2db(X) - mean(pow2db(X)); % att

visualDebug = false;

use_differential = true;

%% Give data the dataframe features structure used in colab

zoh_rmse = [];
lms_rmse = [];

latency = 20; %[1, 5, 10, 15, 20, 25, 30, 35, 40, 50]; % 20
nTaps =  10;   % 3; % [1 10 51 100]; 

t1 = [];
t2 = [];

%  profile on
        
        
for m = 1:length(nTaps)
    for n = 1:length(latency)
    

        df2 = table(wa, 'VariableNames', {'OptPow'});   % Create a table

        if use_differential
            df2.OptPow_diff = [NaN; diff(df2.OptPow)];  % Add the diff steps. 
        else
            df2.OptPow_diff = [(df2.OptPow)];  %%%%%%%%% NO DIFF!
        end
    
        for lag = latency(n):latency(n)+(nTaps(m)-1) 
            df2.(['OptPow_diff_lag' num2str(lag)]) = [NaN(lag, 1); df2.OptPow_diff(1:end-lag)];
        end
        


        % Add target variable
        
        df2.(['OptPow_lag' num2str(latency(n))]) = [NaN(latency(n), 1); df2.OptPow(1:end-latency(n))];   


        if use_differential
            df2.(['OptPow_' num2str(latency(n)) 'stepdiff_target']) = df2.OptPow - df2.(['OptPow_lag' num2str(latency(n))]); 
        else
            df2.(['OptPow_' num2str(latency(n)) 'stepdiff_target']) = df2.OptPow; %%%%%%%%% NO DIFF!
        end

        df2 = rmmissing(df2);
    
    
        %% Organise the training and test data with the same structure.
        
        nTrain = 100000;    
        
        df2_train = df2(1:nTrain, :);
        df2_test = df2(nTrain+1:end, :);
        
        % Create feature column names
        feature_columns = {}; 
        for i = latency(n):latency(n)+(nTaps(m)-1)
            feature_columns{end+1} = ['OptPow_diff_lag' num2str(i)];
        end
        
        % Define target column name
        target_column = ['OptPow_' num2str(latency(n)) 'stepdiff_target'];
    
        % Separate features and target variables for training
        X_train = df2_train(:, feature_columns);
        y_train = df2_train(:, target_column);
        
        xtrain_arr = table2array(X_train);
        ytrain_arr = table2array(y_train);
        
    
        % Separate features and target variables for testing
        X_test = df2_test(:, feature_columns);
        y_test = df2_test(:, target_column);
        
        xtest_arr = table2array(X_test);
        ytest_arr = table2array(y_test);
        
    
        %% LMS TRAINING
%         tic
%         profile on

        % Train Coefficients.
        lock_coeff = false;
        [y_tr,err_tr, wts_tr] = myPytLMS(xtrain_arr, ytrain_arr, latency(n), nTaps(m), [], lock_coeff);
        pred_train = y_tr + df2_train.(['OptPow_lag' num2str(latency(n))]);

%         p1 = profile('info');
%         profile viewer        
%         t1(n) = toc; % end timer
        
        %% LMS TEST/PREDICT
    
%         tic
%         profile on
        
        lock_coeff = true;
        wts = wts_tr(end,:); % Use the last, most updated weights.
        [yt,e, wts] = myPytLMS(xtest_arr, [], latency(n), nTaps(m), wts', lock_coeff);
        
        if use_differential
            predictions_lms = yt + df2_test.(['OptPow_lag' num2str(latency(n))]);
        else
            predictions_lms = yt; %%%%%%%%% NO DIFF!
        end
        
%         p2 = profile('info');
%         profile viewer
%         t2(n) = toc;    
    
         %% LINEAR REGRESSION

        % MATLAB's LR model
        model_lr = fitlm(xtrain_arr, ytrain_arr); % Train    
        predictions_mat = predict(model_lr, xtest_arr); % Test/predict

        % Add back lag to predictions
        if use_differential
            predictions_mat_f = predictions_mat + df2_test.(['OptPow_lag' num2str(latency(n))]); 
        else
            predictions_mat_f = predictions_mat;
        end
    
    
        %% ZOH (zoh) Predictions assumes the prediction is equal to the last lag
    
        predictions_zoh = df2_test.(['OptPow_lag' num2str(latency(n))]);
        
        %% RMSE
    
        lms_rmse(m,n) = myRMSE(df2_test.OptPow, predictions_lms);
    
        zoh_rmse(m,n) = myRMSE(df2_test.OptPow, predictions_zoh);
    
        matLR_rmse(m,n) = myRMSE(df2_test.OptPow, predictions_mat_f);
    
        %% Precompensation
    
        precom_lms = df2_test.OptPow - predictions_lms;
        precom_zoh = df2_test.OptPow - predictions_zoh;
        precom_matlr = df2_test.OptPow - predictions_mat_f;
    
        %% Variance    
        var_lms(m,n) = var(precom_lms);
        var_zoh(m,n) = var(precom_zoh);
        var_matlr(m,n) = var(precom_matlr);
    
        var_input(n) = var(df2_test.OptPow);
    
        %% Rytov variance
        ryt_matlr(m,n) = rytov_vs_latency(db2pow(precom_matlr)); 
        ryt_lms(m,n) = rytov_vs_latency(db2pow(precom_lms));
        ryt_zoh(m,n) = rytov_vs_latency(db2pow(precom_zoh));    
    
        ryt_input(n) = rytov_vs_latency(db2pow(df2_test.OptPow));

        
        %%
        if visualDebug
            try
                close(hFig)
            end
            hFig = figure();
            p_idx = 100000;
            plot(df2_test.OptPow(1:p_idx),DisplayName='Actual Received Signal')
            hold on
            plot(predictions_lms(1:p_idx),DisplayName='LMS prediction')
            plot(predictions_zoh(1:p_idx),DisplayName='zoh prediction')
            plot(predictions_mat_f(1:p_idx),DisplayName='matLR prediction')

            plot(precom_lms(1:p_idx),DisplayName='Precompensated LMS')
            plot(precom_matlr(1:p_idx),DisplayName='Precompensated matLR')
            plot(precom_zoh(1:p_idx),DisplayName='zoh precompensated signal')


            plot(pow2db(X(nTrain+1:nTrain+p_idx)),DisplayName='Raw pow Signal')
            plot((pow2db(X(nTrain+1:nTrain+p_idx)) - predictions_lms(1:p_idx)),DisplayName='Revised pre-compensation')
            yline((mean(pow2db(X))),DisplayName='desired raw pow Signal')
            legend
            grid on
            title([dataset ,' LMS Estimation with ' num2str(latency(n)/Fs*1e3,'%d') ' ms Latency']);
        end

    end
end

% p = profile('info');
% profile viewer

%% PLOT METRICS AGAINST LATENCY

%% Plot RMSE vs. Latency

% figure('Position', [100, 100, 300, 100]),plot(X)
figure
lw = 0.75;
for m = 1:length(nTaps)
    plot(latency,lms_rmse(m,:),'^-', LineWidth = lw, DisplayName='LMS');
    hold on
    plot(latency,zoh_rmse(m,:),'+-', LineWidth = lw, DisplayName='ZOH');
    plot(latency,matLR_rmse(m,:),'*-', LineWidth = lw, DisplayName='LR');
    
    legend('Location', 'southeast');
    xlabel('Estimation Latency [samples]');
    ylabel('RMSE');
    title(['RMSE vs. Latency ' , '[', dataset, ']' ])
    grid on;
end

% I used this to show ntaps in legend
% plot(latency,lms_rmse(m,:),'*-', LineWidth = lw, DisplayName=['LMS mem:  ',num2str(nTaps(m))]);

%% Plot Rytov Variance vs. Latency

figure
plot(latency/10, ryt_input,'--',DisplayName='Input variance');
hold on
for m = 1:length(nTaps)
    plot(latency/10,ryt_lms(m,:),'^-',DisplayName= 'LMS');
    plot(latency/10,ryt_zoh(m,:),'+-',DisplayName= 'ZOH');
    % plot(latency/10,ryt_matlr(m,:),'*-',DisplayName='LR');
    legend('Location', 'northwest');
    xlabel('Estimation Latency [ms]');
    ylabel('Rytov Variance');
    title(['Rytov Variance ' , '[', dataset, ']' ])
    grid on;
end

% plot(latency/10,ryt_matlr(m,:),'^-',DisplayName=['LR. mem:  ',num2str(nTaps(m))]);

%%  Plot Variance vs. Latency

% figure
% plot(latency/10, var_input,'--',DisplayName='Input variance');
% hold on
% for m = 1:length(nTaps)
%     plot(latency*1/Fs*1e3,var_lms(m,:),'*-',DisplayName=['LMS. mem:  ',num2str(nTaps(m))]);
%     plot(latency*1/Fs*1e3,var_zoh(m,:),'+-',DisplayName=['ZOH mem:  ',num2str(nTaps(m))]);
%     plot(latency*1/Fs*1e3,var_matlr(m,:),'^-',DisplayName=['LR. mem:  ',num2str(nTaps(m))]);
%     legend
%     xlabel('Estimation Latency [ms]');
%     ylabel('Variance of Pre-Compensated Waveform');
%     title([' Variance of precompensated signal ', '[', dataset, ']' ])
%     grid on;
% end



















%% Memory sweep!  PLOT METRICS AGAINST MEMORY

%% Plot RMSE vs. nTaps
% 
% figure
% for n = 1:length(latency)
%     plot(nTaps,lms_rmse(:,n),DisplayName=['LMS. Latency:  ',num2str(latency(n))]);
%     hold on
% %      plot(nTaps,matlr_rmse(:,n),DisplayName=['matLR. lat:  ',num2str(latency(n))]);
% %      plot(nTaps,mylr_rmse(:,n),DisplayName=['myLR. lat:  ',num2str(latency(n))]);
% %      plot(nTaps,zoh_rmse(:,n),DisplayName=['zoh. lat:  ',num2str(latency(n))]);
%     legend
%     xlabel('nTaps [samples]');
%     ylabel('RMSE');
%     title(['RMSE vs. nTaps ' , '[', dataset, ']' ])
%     grid on;
% end
% 
% %% Plot Rytov Variance vs. nTaps
% 
% figure
% for n = 1:length(latency)
%     plot(nTaps,ryt_lms(:,n),DisplayName=['LMS. Latency:  ',num2str(latency(n))]);
%     hold on
%      plot(nTaps,ryt_matlr(:,n),DisplayName=['matLR. Latency:  ',num2str(latency(n))]);
% %      plot(nTaps,ryt_mylr(:,n),DisplayName=['myLR. Latency:  ',num2str(latency(n))]);
% %      plot(nTaps,ryt_zoh(:,n),DisplayName=['zoh. Latency:  ',num2str(latency(n))]);
%     legend
%     xlabel('nTaps [samples]');
%     ylabel('Rytov Variance');
%     title(['Rytov Variance vs. nTaps ' , '[', dataset, ']' ])
%     grid on;
% end
% 
% %% Plot Variance vs. nTaps
% 
% figure
% for n = 1:length(latency)
%     plot(nTaps,var_lms(:,n),DisplayName=['LMS. Latency:  ',num2str(latency(n))]);
%     hold on
% %      plot(nTaps,var_matlr(:,n),DisplayName=['matLR. Latency:  ',num2str(latency(n))]);
% %      plot(nTaps,var_mylr(:,n),DisplayName=['myLR. Latency:  ',num2str(latency(n))]);
% %      plot(nTaps,var_zoh(:,n),DisplayName=['zoh. Latency:  ',num2str(latency(n))]);
%     legend
%     xlabel('nTaps [samples]');
%     ylabel('Variance');
%     title(['Variance vs. nTaps ' , '[', dataset, ']' ])
%     grid on;
% end

