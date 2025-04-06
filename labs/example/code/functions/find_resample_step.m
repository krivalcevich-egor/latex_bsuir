function [L, M] = find_resample_step(fs_old, fs_new)
    L = 1; 
    M = 1;
    i = 1;

    while true
        l_tmp = fs_new * i / fs_old;
        
        if mod(l_tmp, 1) == 0
            L = l_tmp;
            M = i;
            break;
        end
        i = i + 1;
    end
end
