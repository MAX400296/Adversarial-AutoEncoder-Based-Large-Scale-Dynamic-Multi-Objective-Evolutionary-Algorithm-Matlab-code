function Problem=TestFunctions(testfunc,dim)
%con=configure();
DEC=dim;
switch testfunc
    case  'DF1'
        Problem.Name    = 'DF1';        % name of test problem
        Problem.NObj    = 2;            % number of objectives
        Problem.XLow    = zeros(DEC,1);  % lower boundary of decision variables, it also defines the number of decision variables
        Problem.XUpp    = ones(DEC,1);   % upper boundary of decision variables
        Problem.FObj    = @DF1;          % Objective function, please read the definition 
end
end


%% test functions

%% test functions
function [F,V] = DF1(X,t)
%% DF1
Fn = 2;
H = 0.75*sinpi(t/2)+1.25;
G = abs(sinpi(t/2));
f1 = X(1);
g = 1+sum((X(2:end)-G).^2);
h = 1-(f1/g)^H;
F = [f1
    g*h];
V = 0.0;
end



