function [z, history] = full_huber_cg(A, b, lambda, rho, alpha)

% Code is adapted from Huber fitting code by Boyd https://web.stanford.edu/~boyd/papers/admm/
% Author: Cassidy Buhler and Hande Benson
% Date last modified: Sept 15th 2021
%
%  [z, history] = full_huber_cg(A, b, lambda, rho, alpha)
% Solves the following problem via CG:
%
%   minimize h( Ax-b ) where h is the Huber loss funciton
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


t_start = tic;

% Global constants and defaults
QUIET    = 0;
MAX_ITER = 10000;
ABSTOL   = 1e-6;
RELTOL   = 1e-4;
gamma    = 1.0;

% Data preprocessing
[m, n] = size(A);

% save a matrix-vector multiply
% Atb = A'*b;

% CG solver
x = 10*ones(n,1);
%f = objective(A, b, lambda, x);
f = 0.0;
c = grad(A, b, lambda, x, m, 1.0);

% cache the factorization
%[L U] = factor(A, rho);

if ~QUIET
    % fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    %  'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

pertcnt = 0;    
nrst = n;
restart = false;

inPowell = false;

for k = 1:MAX_ITER

		xTx = dot(x,x); 
		cTc = dot(c,c);

		% Check for convergence
		if ( sqrt(cTc) <= sqrt(n)*ABSTOL + RELTOL*sqrt(xTx) ) % inftol*inftol*max(1.0, xTx) )
			status = 0;
			break;
        end

		% fprintf("%4d :\t%14.6e\t %14.6e\t | %d \n", k, f, sqrt(cTc/max(1.0, xTx)), pertcnt);

		% Compute step direction
		if ( restart == false )
			dx = -c;
        else
			% Test to see if the Powell restart criterion holds
			if ( (nrst ~= n) && (abs(dot(c, c0)/cTc) > 0.2) )
% fprintf("CUBIT: In Powell!\n");
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
            pTc = dot(pt,c);
			yTc = dot(yt,c);

			u1 = -pTc/ytTyt;
			u2 = 2*pTc/cTyt - yTc/ytTyt;
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
% fprintf("CUBIT: Search direction is not a descent direction.\n");
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

		% Compute the steplength
% 		if ( nrst == 1 ) 
%             alpha0 = 1.0;
%         end
% 		if ( restart <= 1 ) 
%             alpha0 = 1.0/sqrt(cTc);
%         else
%             Adx = A*dx;
%             alpha0 = dot( (b - A*x), Adx )/dot( Adx, Adx );
%             x1 = x + alpha0*dx;
%             numer = sum( dx.*(x1 < -gamma) - lambda*x.*dx.*(-gamma < x1 < gamma) - ...
%                 dx.*(x1 > gamma) );
%             denom = sum( lambda*dx.*dx.*( -gamma < x1 < gamma)  );
%             alpha0 = ( dot( (b - A*x), Adx ) + numer ) / ( dot( Adx, Adx ) + denom );
%         end
%         afind = @(a) dot(dx,grad(A, b, lambda, x + a*dx, n, 1.0));
%         %alpha = fzero( afind, alpha0 );   
%         alpha = alpha0;
        
%         Adx = A*dx;
%         alpha0 = dot( (b - A*x), Adx )/dot( Adx, Adx );
%         x1 = x + alpha0*dx;
%         numer = sum( dx.*(x1 < -gamma) - lambda*x.*dx.*(-gamma < x1 < gamma) - ...
%             dx.*(x1 > gamma) );
%         denom = sum( lambda*dx.*dx.*( -gamma < x1 < gamma)  );
        
     % HANDE: Change the stopping criterion to match ADMM
%         alpha = ( dot( (b - A*x), Adx ) + numer ) / ( dot( Adx, Adx ) + denom );
%         if (alpha < 1e-12) 
%             afind = @(a) objective(A, b, lambda, x + a*dx);
%             alpha = fminbnd(afind, 0, 10);
%             
%             %afind = @(a) dot(dx,grad(A, b, lambda, x + a*dx, n, 1.0));
%             %alpha = fzero( afind, 1.0 );
%         end
        afind = @(a) objective(A, b, lambda, x + a*dx);
        [alpha,fval,exitflag] = fminbnd(afind, 0, 10);

        if (exitflag ~= 1) 
            % fprintf("Line search failed.\n");
            break;
        end
        
        % Take the step and update function value and gradient
		x = x0 + alpha*dx;
%        f = objective(A, b, lambda, x);
        c = grad(A, b, lambda, x, m, 1.0);
%end
%     % x-update
%     q = Atb + rho*(z - u);    % temporary value
%     if( m >= n )    % if skinny
%        x = U \ (L \ q);
%     else            % if fat
%        x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
%     end
% 
%     % z-update with relaxation
%     zold = z;
%     x_hat = alpha*x + (1 - alpha)*zold;
%     z = shrinkage(x_hat + u, lambda/rho);
% 
%     % u-update
%     u = u + (x_hat - z);
% 
%     % diagnostics, reporting, termination checks
%     history.objval(k)  = objective(A, b, lambda, x, z);
% 
%     history.r_norm(k)  = norm(x - z);
%     history.s_norm(k)  = norm(-rho*(z - zold));
% 
%     history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
%     history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
% 
%     if ~QUIET
%         % fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
%             history.r_norm(k), history.eps_pri(k), ...
%             history.s_norm(k), history.eps_dual(k), history.objval(k));
%     end
% 
%     if (history.r_norm(k) < history.eps_pri(k) && ...
%        history.s_norm(k) < history.eps_dual(k))
%          break;
%     end

end

if ~QUIET
    toc(t_start);
end

    z = x;
    history = x;
    fprintf('Iters = %d, inPowell = %d\n', k, inPowell);
end

function p = objective(A, b, lambda, x)
    p =  1/2*huber(A*x - b, 1.0);
end

function d = huber(x, gamma)
    x1 = -x - 0.5;
    x2 = 0.5*x.*x;
    x3 = x - 0.5;
    d = sum(x1.*(x <= -gamma) + x2.*(-gamma < x).*(x < gamma ) + x3.*(x >= gamma));
end

function c = huber_grad(x, n, gamma)
    c = -ones(n,1).*(x <= -gamma) + x.*( -gamma < x).*(x < gamma ) + ones(n,1).*(x >= gamma);
end

function c = grad(A, b, lambda, x, n, gamma)
    c = A'*huber_grad((A*x-b),n,gamma);
end
function af = alphafind(alpha)
    af = grad(A, b, lambda, x + alpha*dx, n)
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end