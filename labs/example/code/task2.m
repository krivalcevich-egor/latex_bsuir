clearvars
clc
addpath('functions')

%% Create signal and Quantized them
N1 = 3;
N2 = 12;
n = 1:200;

s_n = cos(2 * pi * 20 .* n / 12000) + sin(2 * pi * 450 .* n / 12000 - pi / 8);

s_n = s_n / max(abs(s_n));
Quantized_s_n1 = serial_adc(s_n, N1);
Quantized_s_n2 = serial_adc(s_n, N2);