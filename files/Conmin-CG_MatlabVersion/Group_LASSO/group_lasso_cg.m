function [z, history] = groups_lasso_cg(A, b, lambda, p, rho, alpha)
% Code is adapted from group LASSO code by Boyd https://web.stanford.edu/~boyd/papers/admm/
% Author: Cassidy Buhler and Hande Benson
% Date last modified: Sept 15th 2021
%  group_lasso  Solve group lasso problem via CG
% [x, history] = groups_lasso_cg(A, b, p, lambda, rho, alpha);
%
% solves the following problem via CG:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))
%
% The input p is a K-element vector giving the block sizes n_i, so that x_i
% is in R^{n_i}.
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;
% Global constants and defaults
QUIET    = 0;
MAX_ITER = 200;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
gamma    = 1.0;

% Data preprocessing
[m, n] = size(A);

% save a matrix-vector multiply
Atb = A'*b;
% check that sum(p) = total number of elements in x
if (sum(p) ~= n)
    error('invalid partition');
end

% cumulative partition
cum_part = cumsum(p);
% CG solver
% x = zeros(n,1);
x = 0.1*ones(n,1);

%f = objective(A, b, lambda, x);
f = 0.0;
c = grad(A, b, lambda, x, cum_part);

% cache the factorization
%[L U] = factor(A, rho);

if ~QUIET
%    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
%      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

pertcnt = 0;    
nrst = n;
restart = false;

inPowell = false;

for k = 1:MAX_ITER

		xTx = dot(x,x); 
		cTc = dot(c,c);

        %fprintf('%4d :\t%14.6e\t %14.6e\t | %d \n', k, f, sqrt(cTc/max(1.0, xTx)), pertcnt);

		% Check for convergence
		if ( sqrt(cTc) <= sqrt(n)*ABSTOL + RELTOL*sqrt(xTx) ) % inftol*inftol*max(1.0, xTx) )
			status = 0;
			break;
        end

		% Compute step direction
		if ( restart == false )
			dx = -c;
        else
			% Test to see if the Powell restart criterion holds
			if ( (nrst ~= n) && (abs(dot(c, c0)/cTc) > 0.2) )
%fprintf('CUBIT: In Powell!\n');
				nrst = n;
				inPowell = true;
            end

			% If performing a restart, update the Beale restart vectors
			if ( nrst == n )
                pt = alpha*dx;
                yt = c - c0;
                ytTyt = dot(yt,yt);
                cTyt = dot(pt,yt);
            end
            
			p = alpha*dx;
			y = c - c0;

			u1 = -dot(pt,c)/ytTyt;
			u2 = 2*dot(pt,c)/cTyt - dot(yt,c)/ytTyt;
			u3 = cTyt/ytTyt;
			dx = -u3*c - u1*yt - u2*pt;
			
            if ( nrst ~= n )
				u1 = -dot(y,pt)/ytTyt;
				u2 = -dot(y,yt)/ytTyt + 2*dot(y,pt)/cTyt;
				u3 = dot(p,y);
				temp = cTyt/ytTyt*y + u1*yt + u2*pt;
				u4 = dot(temp, y);

				u1 = -dot(p,c)/u3;
                u2 = (u4/u3 + 1)*dot(p,c)/u3 - dot(c,temp)/u3;
                dx = dx - u1*temp - u2*p;
            end
        end

		% Check that the search direction is a descent direction
		dxTc = dot(dx, c);
		if ( dxTc > 0 )
%fprintf('CUBIT: Search direction is not a descent direction.\n');
			status = 3;
			break;
        end

		% Save the current point
		f0 = f;
        x0 = x;
        c0 = c;

		if ( restart == 0 )
			restart = 1;
        else
			if ( nrst == n ) 
                nrst = 0;
            end
			nrst = nrst + 1;
			restart = 2;
        end        
%         Adx = A*dx;
%         alpha0 = dot( (b - A*x), Adx )/dot( Adx, Adx );
%         x1 = x + alpha0*dx;
%         numer = sum( dx.*(x1 < -gamma) - lambda*x.*dx.*(-gamma < x1 < gamma) - ...
%             dx.*(x1 > gamma) );
%         denom = sum( lambda*dx.*dx.*( -gamma < x1 < gamma)  );
%         
     % HANDE: Change the stopping criterion to match ADMM
%         alpha = ( dot( (b - A*x), Adx ) + numer ) / ( dot( Adx, Adx ) + denom );
%         if (alpha < 1e-12) 
%             afind = @(a) objective(A, b, lambda, x + a*dx);
%             alpha = fminbnd(afind, 0, 10);
%             
%             %afind = @(a) dot(dx,grad(A, b, lambda, x + a*dx, n, 1.0));
%             %alpha = fzero( afind, 1.0 );
%         end
        afind = @(a) objective(A, b, lambda, cum_part,x + a*dx,x);
        [alpha,fval,exitflag] = fminbnd(afind, 0, 10);
        if (exitflag ~= 1) 
%            fprintf('Line search failed.\n');
            break;
        end
        
        % Take the step and update function value and gradient
		x = x0 + alpha*dx;
%        f = objective(A, b, lambda, x);
        c = grad(A, b, lambda, x, cum_part);

end

if ~QUIET
    toc(t_start);
end

    z = x;
    history = x;
    fprintf('n = %d, iters = %d. Powell = %d\n', n, k, inPowell);
end

function p = objective(A, b, lambda, cum_part, x, z)
    obj = 0;
    start_ind = 1;
    for i = 1:length(cum_part)
        sel = start_ind:cum_part(i);
        obj = obj + norm(z(sel));
        start_ind = cum_part(i) + 1;
    end
    p = ( 1/2*sum((A*x - b).^2) + lambda*obj );
end

function c = grad(A, b, lambda, x, cum_part)
    start_ind = 1;
    c =A'*(A*x-b);
    for i = 1:length(cum_part)
        sel = start_ind:cum_part(i);
        c(sel) = c(sel) +lambda*x(sel)/(norm(x(sel)));
%          c(sel) = c(sel) +lambda*x(sel)/(norm(x(sel))+0.01);
        start_ind = cum_part(i) + 1;
    end
end
