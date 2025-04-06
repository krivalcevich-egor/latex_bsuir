clearvars
clc

%% Filter options
N = 100; % filter order
Fc_interpolation = 0.1; % fcut
Fc_decimation = 0.1; % fcut

% Calculate filters
bI = fir1(N, Fc_interpolation, 'low', chebwin(N+1)); % Interpolation filter
bD = fir1(N, Fc_decimation, 'low', chebwin(N+1)); % Decimation filter