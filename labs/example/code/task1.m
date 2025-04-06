clearvars
clc
addpath('functions')

%% In this task we need to create 
% a function of serial ADC 
% to harmonical signal

%% 
x = [1.25, -2.5, 0.75, -1.125];
N = 16;

quantized_values = serial_adc(x, N);

disp(quantized_values);
