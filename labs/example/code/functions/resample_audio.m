function [y] = resample_audio(x, fs_old, fs_new, bI, bD)
    [L, M] = find_resample_step(fs_old, fs_new);
    
    xI = upsample(x, L); 
    yI = filter(bI, 1, xI); 
    
    yD = filter(bD, 1, yI); 
    y = downsample(yD, M); 
end