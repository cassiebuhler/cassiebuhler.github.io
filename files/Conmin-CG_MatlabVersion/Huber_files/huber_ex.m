% Huber function fitting example

% Generate problem data
randn('seed', 0);
rand('seed',0);

m = 100000;        % number of examples
n = 5000;       % number of features

for rr=1:100
    fprintf("%d ", rr);
    x0 = randn(n,1);
    A = randn(m,n);
    A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
    b = A*x0 + sqrt(0.01)*randn(m,1);
    b = b + 10*sprand(m,1,200/m);      % add sparse, large noise

    lambda_max = norm( A'*b, 'inf' );
    lambda = 0.001*lambda_max;

    % Solve problem
    [x history] = full_huber_cg(A, b, lambda, 1.0, 1.0);
end