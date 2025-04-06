clearvars
clc
addpath('functions')

N1 = 3;
N2 = 12;
n = 1:200;

s_n = cos(2 * pi * 200 .* n / 12000) + sin(2 * pi * 90 .* n / 12000 - pi / 8);

s_n = s_n / max(abs(s_n));
Quantized_s_n1 = serial_adc(s_n, N1);
Quantized_s_n2 = serial_adc(s_n, N2);

%% Calculate mistake
e_n1 = s_n - Quantized_s_n1;
e_n2 = s_n - Quantized_s_n2;