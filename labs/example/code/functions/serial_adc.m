function [s_q] = serial_adc(x, N)
    s_q = zeros(size(x));
    
    for n = 1:length(x)
        data = x(n); % Initialization data
        BPR = 0; % Initialization BPR (bit-parallel representation)
        
        % Calculate BRP N times
        for i = 1:N
            BPR = BPR + 2^(-i); % Add 2^(-i) to BRP
            
            % Check curry state less then analog signal
            if abs(data) < BPR
                BPR = BPR - 2^(-i); % Sub 2^(-i) to BRP
            end
        end
        
        % Correct sign
        s_q(n) = sign(data) * BPR; 
    end
end

