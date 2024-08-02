%% ASB 2024

% DESCRIPTION: This code is created for the Recurrence quantification
% analysis for movement science workshop for the 2024 American Society of
% Biomechanics Meeting.

% This short tutorial is designed to guide you through using the Recurrence 
% Quantification Analysis (RQA) function. This function is a 'one-stop-shop'
% for multiple versions of RQA including: RQA, CRQA, JRQA, and MDRQA. After
% walking through the input parameter considerations, we will look at an 
% example using step time intervals collect in our lab at UNO. 

%   - Section 1: Prerequisite Calculations
%   - Section 2: Calculate RQA
%   - Section 3: Calculate CRQA
%   - Section 4: Calculate JRQA
%   - Section 5: Calculate MDRQA
%   - Section 5: Change the Parameters


clear; 
clc;

addpath('RQA FUNCTIONS')

%% Section 1: Prerequisite Calculations

% The data being used is a single four minute trial of a healthy young
% adult walking at their preferred pace on an indoor track. The data was
% collected at 200Hz and downsampled by simple decimation to 100Hz and cut
% down to the last 2 minutes of data to reduce processing load. Full data
% can be acquired through the NONAN GaitPrint Database.

load('ASB 2024.mat')

% Each column in the .mat file is the following variables in order: % All
% in tilt forward. time, left contact, right contact, left thigh, right
% thigh, left shank, right shank, left foot, right foot, left upper arm,
% right upper arm, left forearm, right forearm.

% We want to look at right thigh pitch so we will just put that in an
% individual variable to make things easier to follow
r_thigh_pitch = young(1:1000,5);

sampling_rate = 100; % Sampling rate data was collected with

% Average number of frames for one stride. If you are using
% your own data, you should take the time to investigate this on your own.
[~, right_contact_locs] = findpeaks(young(:,3));
stride_time = mean(diff(right_contact_locs));

% One additional input for the RQA function is specifying the minimum
% diagonal length. We found that 60% of the average stride interval was
% sufficient for our data. Specifying the minimum vertical line length is
% less applicable to gait data, but is required for the function so it will
% remain the same.
min_diag = stride_time*0.6;
min_vert = min_diag;

% Calculate time lag:
% The time lag informs us of the necessary amount of delayed values needed
% to reconstruct our data. 
[tau, ~] = AMI_Stergiou(r_thigh_pitch, sampling_rate);   % Average Mutual Information calculation to get the time delay

% Find embedding dimension:
% The embedding dimension informs us of the dimensionality of the data. 
MaxDim = 12;
Rtol = 15;
Atol = 2;
speed = 1;
[dE, ~] = FNN(r_thigh_pitch, tau(1), MaxDim, Rtol, Atol, speed); % False Nearest Neighbors Calculation to obtain the embedding dimension

% Perform state space reconstruction
r_thigh_pitch_psr = psr(r_thigh_pitch, tau(1), dE);

% Plot reconstructed state space in 3D plot
plot3(r_thigh_pitch_psr(:,1), r_thigh_pitch_psr(:,2), r_thigh_pitch_psr(:,3))

%% Section 2: RQA Calculation

% We are estimating RQA by using the pitch of the right thigh, with a
% sample rate of 100Hz, time lag of 11, and dimension of 3 according to our
% prerequisite calculations.

% The RQA function will give us a two outputs: 
%   - 'RP' which is a matrix holding the resulting recurrence plot 
%       &
%   - 'RESULTS' which is a structure holding the following recurrence variables:
%       1.  DIM    - dimension of the input data (used for mdRQA)
%       2.  EMB    - embedding dimension used in the calculation of the
%                    distance matrix
%       3.  DEL    - time lag used in the calculation of the distance
%                    matrix
%       4.  RADIUS - radius used for the recurrence plot
%       5.  NORM   - type of normilization used for the distance matrix
%       6.  ZSCORE - whether or not zscore was used
%       7.  Size   - size of the recurrence plot
%       8.  %REC   - percentage of recurrent points
%       9.  %DET   - percentage of diagonally adjacent recurrent points
%       10. MeanL  - average length of adjacent recurrent points
%       11. MaxL   - maximum length of diagonally adjacent recurrent
%                    points
%       12. EntrL  - Shannon entropy of distribution of diagonal lines
%       13. %LAM   - percentage of vertically adjacent recurrent points
%       14. MeanV  - average length of diagonally adjacent recurrent
%                    points
%       15. MaxV   - maximum length of vertically adjacent recurrent
%                    points
%       16. EntrV  - Shannon entropy of distribution of vertical lines
%       17. EntrW  - Weighted entropy of distribution of vertical
%                    weighted sums
%
% See slide 16.

% We also need to specify several inputs into the RQA function. Below are
% all of the possible input parameters and what they mean.

% RQA FUNCTION INPUTS:
% [RP, RESULTS] = RQA(DATA,TYPE,EMB,DEL,ZSCORE,NORM,SETPARA,SETVALUE,plotOption)
% - DATA = Your given time series

% - TYPE = a string indicating which type of RQA to run (i.e.
%         'RQA', 'cRQA', 'jRQA', 'mdRQA'). The default value is
%         TYPE = 'RQA'.

% - EMB = the number of embedding dimensions (i.e., EMB = 1 would
%        be no embedding via time-delayed surrogates, just using
%        the provided number of colums as dimensions. The default
%        value is EMB = 1.

% - DEL = the delay parameter used for time-delayed embedding (if
%        EMB > 1). The default value is DEL = 1.

% - ZSCORE = indicates, whether the data (i.e., the different
%        columns of DATA, being the different signals or
%        dimensions of a signal) should be z-scored before
%        performing MdRQA:
%        0 - no z-scoring of DATA
%        1 - z-score columns of DATA
%        The default value is ZSCORE = 0.

% - NORM = the type of norm by with the phase-space is normalized.
%        The following norms are available:
%        'euc' - Euclidean distance norm
%        'max' - Maximum distance norm
%        'min' - Minimum distance norm
%        'non' - no normalization of phase-space
%        The default value is NORM = 'non'.

% - SETPARA = the parameter which you would like to set a target
%        value for the recurrence plot (i.e. 'radius' or
%        'recurrence'). The default value is SETPARA = 'radius'.

% - SETVALUE, sets the value of the selected parameter. If
%        SETVALUE = 1, then the radius will be set to 1 if SETPARA
%        = 'radius' or the radius will be adjusted until the
%        recurrence is equal to 1 if SETPARA = 'recurrence'. The
%        default value if SETPARA = 'radius' is 1. The default
%        value if SETPARA = 'recurrence' is 2.5.

% - plotOption, is a boolean where if true the recurrence plot will
%        be created and displayed.

% Now we will run RQA
% - Note that we are finding the optimal radius using a percent recurrence
% of 2.5%. More information on choosing a proper radius can be found on 
% slide 12
plotOption = 1;
[RP, RQA_output] = RQA_072424(r_thigh_pitch, 'RQA', tau(1), dE, 0, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

% Lets do the same for the older adult trial. Notice any differences?
r_thigh_pitch = old(1:1000,5);
sampling_rate = 100;
[~, right_contact_locs] = findpeaks(old(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;
[tau, ~] = AMI_Stergiou(r_thigh_pitch, sampling_rate);
[dE, ~] = FNN(r_thigh_pitch, tau(1), MaxDim, Rtol, Atol, speed);
plotOption = 1;
[RP, RQA_output] = RQA_072424(r_thigh_pitch, 'RQA', tau(1), dE, 0, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

%%   - Section 3: Calculate CRQA

% Now let's look at the inter-limb coordination between the right thigh and 
% left thigh using Cross Recurrence Quantification Analysis
r_thigh_pitch = young(1:1000,5);
l_thigh_pitch = young(1:1000,4);

% Here, we have to find the proper time delay and embedding dimension for 
% both thighs
[r_tau_thigh, ~] = AMI_Stergiou(r_thigh_pitch, sampling_rate);   % Finding the time delay for the right thigh
[l_tau_thigh, ~] = AMI_Stergiou(l_thigh_pitch, sampling_rate);   % Finding the time delay for the left thigh

[r_dE_thigh, ~] = FNN(r_thigh_pitch, r_tau_thigh(1), MaxDim, Rtol, Atol, speed);   % Finding the embedding dimension for the right thigh
[l_dE_thigh, ~] = FNN(l_thigh_pitch, l_tau_thigh(1), MaxDim, Rtol, Atol, speed);    % Finding the embedding dimension for the left thigh

% With two different time delays and embedding dimensions, we must choose
% which one we want to use. The most common procedure is to use the largest
% value obtained. See slide 29.
% Below is a logical that chooses our highest tau and
% embedding dimension.

% Choosing largest time delay between both systems
if r_tau_thigh(1) > l_tau_thigh(1)
    chosenTau = r_tau_thigh(1);
else
    chosenTau = l_tau_thigh(1);
end

% Choosing largest embedding dimensions between both systems
if r_dE_thigh > l_dE_thigh
    chosenDim = r_dE_thigh;
else
    chosenDim = l_dE_thigh;
end

% Keep min values the same
[~, right_contact_locs] = findpeaks(young(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;

% RUNNING CRQA
% Now we must specify in our TYPE input that we want 'cRQA'
% Also note that our we have both data columns now input as our time series
plotOption = 1;
[RP, RQA_output] = RQA_072424([r_thigh_pitch, l_thigh_pitch], 'cRQA', chosenTau, chosenDim, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

% Lets do the same for the older adult trial. Notice any differences?
r_thigh_pitch = old(1:1000,5);
l_thigh_pitch = old(1:1000,4);

[r_tau_thigh, ~] = AMI_Stergiou(r_thigh_pitch, sampling_rate);
[l_tau_thigh, ~] = AMI_Stergiou(l_thigh_pitch, sampling_rate);
[r_dE_thigh, ~] = FNN(r_thigh_pitch, r_tau_thigh(1), MaxDim, Rtol, Atol, speed);
[l_dE_thigh, ~] = FNN(l_thigh_pitch, l_tau_thigh(1), MaxDim, Rtol, Atol, speed);

if r_tau_thigh(1) > l_tau_thigh(1)
    chosenTau = r_tau_thigh(1);
else
    chosenTau = l_tau_thigh(1);
end

if r_dE_thigh > l_dE_thigh
    chosenDim = r_dE_thigh;
else
    chosenDim = l_dE_thigh;
end

[~, right_contact_locs] = findpeaks(old(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;

plotOption = 1;
[RP, RQA_output] = RQA_072424([r_thigh_pitch, l_thigh_pitch], 'cRQA', chosenTau, chosenDim, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

%%   - Section 4: Calculate JRQA

% Now let's look at the coordination of temporal recurrences between the
% right thigh and left thigh using Joint Recurrence Quantification Analysis 

% RUNNING JRQA
% Now we must specify in our TYPE input that we want 'jRQA'
% The current RQA function does not yet handle two embedding
% dimensions and two time delays for each system. For now, we will use the 
% largest tau and embedding dimension as before. See slide 45.

r_thigh_pitch = young(1:1000,5);
l_thigh_pitch = young(1:1000,4);

[r_tau_thigh, ~] = AMI_Stergiou(r_thigh_pitch, sampling_rate);
[l_tau_thigh_pitch, ~] = AMI_Stergiou(l_thigh_pitch, sampling_rate);

[r_dE_thigh, ~] = FNN(r_thigh_pitch, r_tau_thigh(1), MaxDim, Rtol, Atol, speed);
[l_dE_thigh_pitch, ~] = FNN(l_thigh_pitch, l_tau_thigh_pitch(1), MaxDim, Rtol, Atol, speed);

if r_tau_thigh(1) > l_tau_thigh_pitch(1)
    chosenTau = r_tau_thigh(1);
else
    chosenTau = l_tau_thigh_pitch(1);
end

if r_dE_thigh > l_dE_thigh_pitch
    chosenDim = r_dE_thigh;
else
    chosenDim = l_dE_thigh_pitch;
end

[~, right_contact_locs] = findpeaks(young(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;

plotOption = 1;
[RP, RQA_output] = RQA_072424([r_thigh_pitch, l_thigh_pitch], 'jRQA', chosenTau, chosenDim, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

% Lets do the same for the older adult trial. Notice any differences?
r_thigh_pitch = old(1:1000,5);
l_thigh_pitch = old(1:1000,4);

[r_tau_thigh, ~] = AMI_Stergiou(r_thigh_pitch, sampling_rate);
[l_tau_thigh_pitch, ~] = AMI_Stergiou(l_thigh_pitch, sampling_rate);

[r_dE_thigh, ~] = FNN(r_thigh_pitch, r_tau_thigh(1), MaxDim, Rtol, Atol, speed);
[l_dE_thigh_pitch, ~] = FNN(l_thigh_pitch, l_tau_thigh_pitch(1), MaxDim, Rtol, Atol, speed);

if r_tau_thigh(1) > l_tau_thigh_pitch(1)
    chosenTau = r_tau_thigh(1);
else
    chosenTau = l_tau_thigh_pitch(1);
end

if r_dE_thigh > l_dE_thigh_pitch
    chosenDim = r_dE_thigh;
else
    chosenDim = l_dE_thigh_pitch;
end

[~, right_contact_locs] = findpeaks(old(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;

plotOption = 1;
[RP, RQA_output] = RQA_072424([r_thigh_pitch, l_thigh_pitch], 'jRQA', chosenTau, chosenDim, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

%%   - Section 5: Calculate MDRQA

% Now let's look at the application of Multidimensional Recurrence
% Quantification Analysis

% Here we will only use young adult walking data for the right thigh, right
% shank, right upper arm, and left upper arm.
r_thigh_pitch = young(1:1000,5);
l_thigh_pitch = young(1:1000,4);
r_upper_arm = young(1:1000,11);
l_upper_arm = young(1:1000,10);
dat_md_young = [r_thigh_pitch l_thigh_pitch r_upper_arm l_upper_arm];

[~, right_contact_locs] = findpeaks(young(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;

% First, we will run MdRQA without time delayed embedding. Then, we will
% also run MdRQA with time delayed embedding based on the parameter
% estimation method proposed by Wallot and Monster 2018.

% RUNNING MdRQA without time delayed embedding
% Now we must specify in our TYPE input that we want 'MdRQA'.
% For MdRQA without time delayed embedding, we do not need to specify any
% values for time delays and embedding dimensions. So, we will just put 1
% for both parameters. See slide 55. Since the magnitudes of each time
% series were differing, we will set ZSCORE input as 1 so that the function 
% rescales the time series prior to MdRQA. See slide 53.
plotOption = 1;
[RP, RQA_output] = RQA_072424(dat_md_young, 'mdRQA', 1, 1, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

% Lets do the same for the older adult trial. Notice any differences?
r_thigh_pitch = old(1:1000,5);
l_thigh_pitch = old(1:1000,4);
r_upper_arm = old(1:1000,11);
l_upper_arm = old(1:1000,10);
dat_md_old = [r_thigh_pitch l_thigh_pitch r_upper_arm l_upper_arm];

[~, right_contact_locs] = findpeaks(old(:,3));
stride_time = mean(diff(right_contact_locs));
min_diag = stride_time*0.6;
min_vert = min_diag;

plotOption = 1;
[RP, RQA_output] = RQA_072424(dat_md_old, 'mdRQA', 1, 1, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

% To run MdRQA with time delayed embedding, we have to find the proper time
% delay and embedding dimension for the multivariate time series. See slide
% 56~64.
tau = mdDelay(dat_md_young, 'maxLag', 150, 'criterion', 'localMin', 'plottype', 'all', 'minAMI', true); % Finding the average time delay of all variables
tau = round(tau); % Rounding the average time delay
[fnnPerc, embTimes, dim] = mdFnn(dat_md_young, tau, 'maxEmb', 20); % Finding the embedding dimension for the multivariate time series

plotOption = 1;

% RUNNING MdRQA with time delayed embedding
% Again, we must specify in our TYPE input that we want 'MdRQA'.
% This time we will specify our EMB input and DEL input as dim and tau that
% is found above, respectively.
[rp_embedded, result_embedded] = RQA_072424(dat_md_young, 'mdRQA', tau, dim, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);

% Lets do the same for the older adult trial. Notice any differences?
tau = mdDelay(dat_md_old, 'maxLag', 150, 'criterion', 'localMin', 'plottype', 'all', 'minAMI', true);
tau = round(tau);
[fnnPerc, embTimes, dim] = mdFnn(dat_md_old, tau, 'maxEmb', 20);

plotOption = 1;
[rp_embedded, result_embedded] = RQA_072424(dat_md_old, 'mdRQA', tau, dim, 1, 'NON', 'recurrence', 2.5, min_diag, min_vert, plotOption);
