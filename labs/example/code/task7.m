clearvars
clc
addpath('functions')

%% Make dir 
output_dir = 'wav/output';
input_dir = 'wav';

%% Check audio file
[x, fs_old] = audioread(fullfile(input_dir, 'FA06_01.wav'));
fs_new = 32000;
N = 100;

% Calculate L and aM
[L, M] = find_resample_step(fs_old, fs_new);
int_coef = 1 / L;
dec_coef = 1 / M; 
if int_coef >= 1
    int_coef = int_coef - 0.000000000000001;
end
if dec_coef >= 1
    dec_coef = dec_coef - 0.000000000000001;
end

% Interpolation filter
bI = fir1(N-1, int_coef, 'low', chebwin(N, 60)); 

% Decimation filter
bD = fir1(N-1, dec_coef, 'low', chebwin(N, 60)); 

% resample
y = resample_audio(x, fs_old, fs_new, bI, bD);

% save result
audiowrite(fullfile(output_dir, 'output_audio_t7.wav'), y, fs_new);
