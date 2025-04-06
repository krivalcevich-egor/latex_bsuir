clearvars
clc
addpath('functions')

figure;
% Plot spectrogram
subplot(2,2,1);
specgram(x, 512, fs_old,hann(512),475);
title('Spectrogram of Original Signal');
set(gca,'Clim', [-65 -15]);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 14);

subplot(2,2,2);
specgram(y, 512, fs_new,hann(512),475);
title('Spectrogram of Resampled Signal');
set(gca,'Clim', [-65 -15]);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 14);

% Plot time-domain representation
subplot(2,2,3);
plot((1:length(x))/fs_old, x);
xlim([0, length(x)/fs_old]);
title('Original Signal', 'FontName', 'Times New Roman', 'FontSize', 14);
xlabel('Time (s)');
ylabel('Amplitude');
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 14);

subplot(2,2,4);
plot((1:length(y))/fs_new, y);
xlim([0, length(y)/fs_new]);
title('Resampled Signal');
xlabel('Time (s)');
ylabel('Amplitude');
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 14);