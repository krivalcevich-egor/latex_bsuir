clearvars
clc
addpath('functions')

%% In this task we need to create
%  a function for calculation of decimation 
% and interpolation parameters for
% converting the sampling rate of an audio 
% signal with a non-integer step

%% 
fs_old = 44100; 
fs_new = 48000; 

[L, M] = find_resample_step(fs_old, fs_new);
fprintf('Interpolation coefficientL = %d\n', L);
fprintf('Decimation coefficient M = %d\n', M);

fs_old = 48000; 
fs_new = 14000; 

[L, M] = find_resample_step(fs_old, fs_new);
fprintf('Interpolation coefficient L = %d\n', L);
fprintf('Decimation coefficient M = %d\n', M);
