% Huber function fitting example

% Generate problem data
randn('seed', 0);
rand('seed',0);

m = 100000;        % number of examples
n = 5000;       % number of features

for rr=1:100
    x0 = randn(n,1);
    A = randn(m,n);
    A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
    b = A*x0 + sqrt(0.01)*randn(m,1);
    b = b + 10*sprand(m,1,200/m);      % add sparse, large noise

    % Solve problem
    [x history] = huber_admm(A, b, 1.0, 1.0);

end

