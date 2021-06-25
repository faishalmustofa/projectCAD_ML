function [PCG_resampled, data_S1, data_Systole, data_S2, data_Diastole] = challenge(PCG, Fs1)
%
% Sample entry for the 2016 PhysioNet/CinC Challenge.
%
% INPUTS:
% recordName: string specifying the record name to process
%
% OUTPUTS:
% classifyResult: integer value where
%                     1 = abnormal recording
%                    -1 = normal recording
%                     0 = unsure (too noisy)
%
% To run your entry on the entire training set in a format that is
% compatible with PhysioNet's scoring enviroment, run the script
% generateValidationSet.m
%
% The challenge function requires that you have downloaded the challenge
% data 'training_set' in a subdirectory of the current directory.
%    http://physionet.org/physiobank/database/challenge/2016/
%
% This dataset is used by the generateValidationSet.m script to create
% the annotations on your training set that will be used to verify that
% your entry works properly in the PhysioNet testing environment.
%
%
% Version 1.0
%
%
% Written by: Chengyu Liu, Fubruary 21 2016
%             chengyu.liu@emory.edu
%
% Last modified by:
%
%

%% Load the trained parameter matrices for Springer's HSMM model.
% The parameters were trained using 409 heart sounds from MIT heart
% sound database, i.e., recordings a0001-a0409.
load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
load('parms_cnn.mat');
load('learned_parms.mat')
parms.maxpooling = 2;

N=60; sr = 1000; 
Wn = 45*2/sr; 
b1 = fir1(N,Wn,'low',hamming(N+1));
Wn = [45*2/sr, 80*2/sr];
b2 = fir1(N,Wn,hamming(N+1));
Wn = [80*2/sr, 200*2/sr];
b3 = fir1(N,Wn,hamming(N+1));
Wn = 200*2/sr;
b4 = fir1(N,Wn,'high',hamming(N+1));

%% Load data and resample data
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;
%[PCG, Fs1, nbits1] = wavread([recordName '.wav']);  % load data
%[PCG,Fs1] = audioread([recordName '.wav']);  % load data

if length(PCG)>60*Fs1
    PCG = PCG(1:60*Fs1);
end
            
% resample to 1000 Hz
PCG_resampled = resample(PCG,springer_options.audio_Fs,double(Fs1)); % resample to springer_options.audio_Fs (1000 Hz)
% filter the signal between 25 to 400 Hz
PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
% remove spikes
PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);

%% Running runSpringerSegmentationAlgorithm.m to obtain the assigned_states
assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,... 
                springer_options.audio_Fs,... 
                Springer_B_matrix, Springer_pi_vector,...
                Springer_total_obs_distribution, false);

% get states
idx_states = get_states(assigned_states);

features_time = get_features_time(PCG_resampled,idx_states);
features_freq = get_features_frequency(PCG_resampled,idx_states);
%%wavelet_features = get_wavelet_features(PCG_resampled,idx_states);
features = [features_time, features_freq];

data_S1 = double.empty();
data_Systole = double.empty();
data_S2 = double.empty();
data_Diastole = double.empty();
for i=1:size(idx_states,1)-1
    S1 = PCG_resampled(idx_states(i,1):idx_states(i,2));
    Systole = PCG_resampled(idx_states(i,2):idx_states(i,3));
    S2 = PCG_resampled(idx_states(i,3):idx_states(i,4));
    Diastole = PCG_resampled(idx_states(i,4):idx_states(i+1,1));
    if (size(data_S1) == 0)
        data_S1 = {S1};
        data_Systole = {Systole};
        data_S2 = {S2};
        data_Diastole = {Diastole};
    else
        data_S1 = [data_S1, {S1}];
        data_Systole = [data_Systole, {Systole}];
        data_S2 = [data_S2, {S2}];
        data_Diastole = [data_Diastole, {Diastole}];
    end
end
%%
function idx_states = get_states(assigned_states)
    indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

    if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end

    indx2                = indx(K:end);
    rem                  = mod(length(indx2),4);
    indx2(end-rem+1:end) = [];
    idx_states           = reshape(indx2,4,length(indx2)/4)';

%%
%%
function features = get_features_time(PCG,idx_states)
    %% Feature calculation
    m_RR        = round(mean(diff(idx_states(:,1))));             % mean value of RR intervals
    sd_RR       = round(std(diff(idx_states(:,1))));              % standard deviation (SD) value of RR intervals
    mean_Cardiac_cycle  = round(mean(idx_states(:,1)-idx_states(:,4)));             % mean value of Cardiac_cycle
    sd_Cardiac_cycle    = round(std(idx_states(:,1)-idx_states(:,4)));             % SD value of Cardiac_cycle

    for i=1:size(idx_states,1)-1
        R_SysRR(i)  = (idx_states(i,3)-idx_states(i,2))/(idx_states(i+1,1)-idx_states(i,1))*100;
        R_DiaRR(i)  = (idx_states(i+1,1)-idx_states(i,4))/(idx_states(i+1,1)-idx_states(i,1))*100;
        R_SysDia(i) = R_SysRR(i)/R_DiaRR(i)*100;

        %skewness
        SK_Cardiac_cycle(i)  = skewness(PCG(idx_states(i,1):idx_states(i+1,1)));

        % kurtosis
        KU_Cardiac_cycle(i)  = kurtosis(PCG(idx_states(i,1):idx_states(i+1,1)));

        P_Cardiac_cycle(i)     = sum(abs(PCG(idx_states(i,1):idx_states(i+1,1))))/(idx_states(i+1,1)-idx_states(i,1));

    end

    m_Ratio_SysRR   = mean(R_SysRR);  % mean value of the interval ratios between systole and RR in each heart beat
    sd_Ratio_SysRR  = std(R_SysRR);   % SD value of the interval ratios between systole and RR in each heart beat
    m_Ratio_DiaRR   = mean(R_DiaRR);  % mean value of the interval ratios between diastole and RR in each heart beat
    sd_Ratio_DiaRR  = std(R_DiaRR);   % SD value of the interval ratios between diastole and RR in each heart beat
    m_Ratio_SysDia  = mean(R_SysDia); % mean value of the interval ratios between systole and diastole in each heart beat
    sd_Ratio_SysDia = std(R_SysDia);  % SD value of the interval ratios between systole and diastole in each heart beat

    mSK_Cardiac_cycle = mean(SK_Cardiac_cycle);
    sdSK_Cardiac_cycle = std(SK_Cardiac_cycle);

    mKU_Cardiac_cycle = mean(KU_Cardiac_cycle);
    sdKU_Cardiac_cycle = std(KU_Cardiac_cycle);

    features = [m_RR, sd_RR, mean_Cardiac_cycle, sd_Cardiac_cycle, SK_Cardiac_cycle,KU_Cardiac_cycle, ...
                P_Cardiac_cycle,m_Ratio_SysRR,sd_Ratio_SysRR,m_Ratio_DiaRR,sd_Ratio_DiaRR, ...
                m_Ratio_SysDia,sd_Ratio_SysDia,mSK_Cardiac_cycle,sdSK_Cardiac_cycle,mKU_Cardiac_cycle, sdKU_Cardiac_cycle];

%%
%%
function features = get_features_frequency(PCG,idx_states)
    NFFT = 256;
    f = (0:NFFT/2-1)/(NFFT/2)*500;
    freq_range = [25,45;45,65;65,85;85,105;105,125;125,150;150,200;200,300;300,500];
    p_Cardiac_cycle  = nan(size(idx_states,1)-1,NFFT/2);
    for row=1:size(idx_states,1)-1
        cardiac_cycle = PCG(idx_states(row,1):idx_states(row+1,1));
        cardiac_cycle = cardiac_cycle.*hamming(length(cardiac_cycle));
        Ft = fft(cardiac_cycle,NFFT);
        p_Cardiac_cycle(row,:) = abs(Ft(1:NFFT/2));
    end
    P_Cardiac_cycle = nan(1,size(freq_range,1));
    for bin=1:size(freq_range,1)
        idx = (f>=freq_range(bin,1)) & (f<freq_range(bin,2));
        P_Cardiac_cycle(1,bin) = median(median(p_Cardiac_cycle(:,idx)));
    end
    features = [P_Cardiac_cycle];
