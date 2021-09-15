function [z, history] = full_huber_cubic(A, b, lambda, rho, alpha)
% Code is adapted from Huber fitting code by Boyd https://web.stanford.edu/~boyd/papers/admm/
% Author: Cassidy Buhler and Hande Benson
% Date last modified: Sept 15th 2021% lasso  Solve lasso problem via CG
%
% [z, history] =  full_huber_cubic(A, b, lambda, rho, alpha);
%
% Solves the following problem via CG with cubic regularization:
%
%   minimize h( Ax-b ) where h is the Huber loss funciton
% The solution is returned in the vector x.
%
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

t_start = tic;

% Global constants and defaults
QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
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
restart = 0;

inPowell = false;
dprat = 0.0;
ddprat = 0.0;
lam = 0;
k = 0;

while (k < MAX_ITER)
    
        k = k + 1;

        xTx = dot(x,x); 
		cTc = dot(c,c);

		% Check for convergence
		if ( sqrt(cTc) <= sqrt(n)*ABSTOL + RELTOL*sqrt(xTx) ) % inftol*inftol*max(1.0, xTx) )
			status = 0;
			break;
        end

		% Compute step direction
		if ( restart == 0 )
            % fprintf("%4d :\t%14.6e\t %14.6e\t | %d \n", k, f, sqrt(cTc/max(1.0, xTx)), pertcnt);
			dx = -c;
        else
			% Test to see if the Powell restart criterion holds
			if ( abs(dot(c,c0)/cTc) > 0.2 && restart > 1 && nrst ~= n )
                if (nrst == 0)
                    nrst = n;
                end
                inPowell = true;
                % fprintf("Here for this\n");
                if ( lam == 0.0 || lam > 1e6)
                    if ( dprat0 ~= 0.0 ) 
                        ddprat = ( abs(dot(c,c0)/cTc) - 2*prat + prat0 ) / ( (lam - olambda)*(oolambda-olambda) );
                    end
                    dprat0 = dprat;
                    if ( lam > 0 ) 
                        dprat = ( abs(dot(c,c0)/cTc) - prat ) / ( lam - olambda );
                    end
                    if (exist('prat','var'))
                        prat0 = prat;
                    end
                    prat = abs(dot(c,c0)/cTc);
                    alpha = alpha0;
                    restart = restart0;
                    x = x0;
                    dx = dx0;
                    c = c0;
                    c0 = c00;
                    % c = grad(A, b, lambda, x, m, 1.0);
                    xTx = dot(x,x);
                    cTc = dot(c,c);
                    if (exist('olambda','var'))
                        oolambda = olambda;
                    end
                    olambda = lam;
                    if ( lam == 0.0 )
                        lam = prat/0.2; 
                    else
                        if ( pertcnt <= 2 )
                            lam = olambda - prat/dprat;
                        else
                            if ( dprat*dprat - 2*ddprat*prat > 0 && abs(ddprat) > 1e-8 )
                                lam = olambda + max( (-dprat + sqrt(dprat*dprat - 2*ddprat*prat))/ddprat, ...
                                            (-dprat - sqrt(dprat*dprat - 2*ddprat*prat))/ddprat ); 
                            else
                                lam = 2*olambda;
                            end
                        end
                        if ( lam < 1e-12 ) 
                            lam = 2*olambda;
                        end
                    end
                else
                    lam = 2*lam;
                end
                pertcnt = pertcnt + 1;
                k = k - 1;
            else
				dx0 = dx;
				% lam = 0.0;
                if ( pertcnt > 0 )
                    lam = lam/2.0;
                else
                    lam = 0.0;
                end
				dprat = 0.0;
				dprat0 = 0.0;
				prat0 = 0.0;
				% fprintf("%4d :\t%14.6e\t %14.6e\t | %d \n", k, f, sqrt(cTc/max(1.0, xTx)), pertcnt);
				pertcnt = 0;
                
            end

			% If performing a restart, update the Beale restart vectors
			if ( nrst == n )
                pt = alpha*dx;
                yt = c - c0;
                ytTyt = dot(yt,yt);
                cTyt = dot(pt,yt);
                cTct = dot(pt,pt);
            end
            
            p = alpha*dx;
            y = c - c0;
            pTc = dot(pt,c);
            yTc = dot(yt,c);

            u1 = -pTc/ytTyt;
            u2 = 2*pTc/cTyt - yTc/ytTyt;
            u3 = cTyt/ytTyt;
            
            if ( pertcnt == 0 ) 
                dx = -u3*c - u1*yt - u2*pt;
            else
                bracket = lam*lam + 2*lam*ytTyt/cTyt + ytTyt/cTct;
                a = -cTyt/(lam*cTyt + ytTyt);
                b1 = (-lam*cTyt - 2*ytTyt)*ytTyt/(cTyt*cTct*bracket*(lam*cTyt + ytTyt));
                d = lam/(bracket*(lam*cTyt + ytTyt));
                e = ytTyt/(cTct*bracket*(lam*cTyt + ytTyt));

                bracket = lam*lam+lam/u3+lam*ytTyt/cTyt+cTyt/(u3*cTct);
                denom = u3*u3*cTct*lam*bracket + u3*cTct*bracket;
                d = u3*u3*lam*(cTct/cTyt) / denom;

                dx = a*c + b1*pTc*pt + d*yTc*yt + e*yTc*pt + e*pTc*yt;
            end
			
            if ( nrst ~= n )
                if ( pertcnt == 0 )
                    u1 = -dot(y,pt)/ytTyt;
                    u2 = -dot(y,yt)/ytTyt + 2*dot(y,pt)/cTyt;
                    u3 = dot(p,y);
                    temp = cTyt/ytTyt*y + u1*yt + u2*pt;
                    u4 = dot(temp, y);
                    
                    u1 = -dot(p,c)/u3;
                    u2 = (u4/u3 + 1)*dot(p,c)/u3 - dot(c,temp)/u3;
                    dx = dx - u1*temp - u2*p;
                else
                    % inv(H+lI)*y
                    ptTy = dot(pt,y);
                    ytTy = dot(yt,y);
                    temp1 = -(a*y + b1*ptTy*pt + d*ytTy*yt + e*ytTy*pt + e*ptTy*yt);
                    
                    % inv(H+lI)*invH*p
                    a2 = 1.0/(lam*u3+1.0);
                    b2 = lam*b1;
                    d2 = lam*d;
                    e2 = lam*e;
                    ptTp = dot(pt,p);
                    ytTp = dot(yt,p);
                    temp2 = a2*p + b2*ptTp*pt + d2*ytTp*yt + e2*ytTp*pt + e2*ptTp*yt;
                    
                    u10 = dot(temp1, c); % yT inv(H+lI) g
                    u11 = dot(temp2, c); % pT invH inv(H+lI) g
                    u12 = dot(temp1, y); % yT inv(H+lI) y
                    u13 = dot(temp2, y); % pT invH inv(H+lI) y
                    u14 = dot(temp2, p); % pT invH inv(H+lI) p
                    u15 = dot(p, y);	 % pTy
                    
                    denom = lam*u14*(u15 + u12) + u13*u13;
                    
                    % dx = -inv(H+lI)*g --- Remember the minus sign
                    dx = dx + lam*u14*u10*temp1/denom - ...
                        (u15 + u12)*u11*temp2/denom + ...
                        u13*u10*temp2/denom + u13*u11*temp1/denom;
                end
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
        if (exist('c0','var'))
            c00 = c0;
        end
        c0 = c;
        alpha0 = alpha;
		restart0 = restart;

		if ( restart == 0 )
			restart = 1;
        else
            restart = 2;
			if ( pertcnt == 0 )
				if ( nrst == n ) 
                    nrst = 0;
                end
				nrst = nrst + 1;
				restart = 2;
            end
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
%         alpha = ( dot( (b - A*x), Adx ) + numer ) / ( dot( Adx, Adx ) + denom );
        
        afind = @(a) objective(A, b, lambda, x + a*dx);
        alpha = fminbnd(afind, 0, 10);
        
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
    fprintf('Iters = %d, inCubic = %d\n', k, inPowell);

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