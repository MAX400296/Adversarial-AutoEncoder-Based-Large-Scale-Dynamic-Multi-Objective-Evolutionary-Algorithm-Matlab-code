function [Population,FEs] = LMOEAInitialization(Probelm,t,CostFunction,FEs,initPop)
%Initialization - Generate multiple initial solutions.
%
%   P = obj.Initialization() randomly generates the decision
%   variables of obj.N solutions and returns the SOLUTION objects.
%
%   P = obj.Initialization(N) generates N solutions.
%
%   This function is usually called at the beginning of algorithms.
%
%   Example:
%       Population = Problem.Initialization()
if nargin == 5  
    if size(initPop,2)<Probelm.N 
        n1 = Probelm.N - size(initPop,2);
        PopDec1 = zeros(n1,Probelm.D);
        PopDec1(:,:) = unifrnd(repmat(Probelm.lower(),n1,1),repmat(Probelm.upper(),n1,1));
        initPop = cat(1,initPop',PopDec1);
        PopDec = initPop;
        N = size(PopDec,1);
    else
        PopDec = initPop';
        N = size(PopDec,1);
    end
else
    N = Probelm.N;
    PopDec = zeros(N,Probelm.D);
    PopDec(:,:) = unifrnd(repmat(Probelm.lower(),N,1),repmat(Probelm.upper(),N,1));
end
for i = 1:N
    X(i,:) = PopDec(i,:);
    Y(i,:) = CostFunction(X(i,:),t)';
    FEs = FEs+1;
end
Population = SOLUTION(X,Y);
end