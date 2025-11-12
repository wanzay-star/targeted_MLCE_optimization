
% This code replicates the LMS method used in the google Colab code.
% It allows comparison of LMS results obtained in MATLAB with those
% obtained in Colab with Python.

function [y,e, wts] = myPytLMS(x, d, nSteps, N, w, lock_coeff)

    % Inputs:
    % x - Input signal
    % d - Desired signal
    % N - Filter order
    % w - Initial pre-trained weights (for locked coefficients)
    % lock_coeff - Boolean flag to lock weights (true = no update)

    L = length(x);
    
    % Initialize outputs
    if isempty(w)
        w = zeros(N, 1); % Default: Initialize weights to zero
        % w = zeros(1,N);
    end

    % e = zeros(L-nSteps, 1); % Error vector
    e = zeros(L, 1);       
    y = zeros(L, 1);       % Output vector
    wts = zeros(L , N); 
    % wts = zeros(L - N - nSteps + 1, N); % Preallocate weight storage
    mu = 1e-5;             % Learning rate
    epsilon = 1e-6;        % Small constant to prevent division by zero
    
    % LMS Algorithm Iteration
    c = 1; % Countera

    % for n = N:L-nSteps
    for n = 1:length(x)
        % x_n = x(n-N+1:n);
        % x_n = x(n,:)';
        x_n = flip(x(n,:))';
        
        % Calculate predicted output
        % y(n+nSteps) = w * x_n;
        y(n) = w' * x_n;
        
        % Calculate error
        if ~isempty(d)
            e(n) = d(n) - y(n);
        end
        % e(n) = x(n+nSteps) - y(n+nSteps);

        % Update weights if lock_coeff is false
        if ~lock_coeff
            norm_factor = 1;
            w = w + (mu / norm_factor) * e(n) * x_n;
        end
        
        % Store weights for analysis
        wts(c, :) = w;
        c = c + 1;
    end

end
