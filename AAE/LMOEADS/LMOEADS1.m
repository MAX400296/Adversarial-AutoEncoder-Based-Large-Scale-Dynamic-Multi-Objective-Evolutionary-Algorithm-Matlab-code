%The code is from the PlatEMO platform.
%https://ww2.mathworks.cn/matlabcentral/fileexchange/105260-platemo
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
function  [PopX,Pareto,POF_iter,Population] = LMOEADS1( ProblemNew,popSize,MaxIt, t, initPop)
%% Parameter settings
Ns = 30; 
M = ProblemNew.NObj; 
Nw = M+10; 
N = popSize; 
FEs = ProblemNew.FEs;
maxFEs = MaxIt;
Problem.M = ProblemNew.NObj; %1*1double   
Problem.upper = (ProblemNew.XUpp)'; % 1*100double  
Problem.lower = (ProblemNew.XLow)'; % 1*100double  
Problem.D = size(ProblemNew.XLow,1);  % 1*1double  
Problem.Evaluation = ProblemNew.FObj;  
Problem.Name = ProblemNew.Name;
Problem.N = popSize; 
CostFunction = Problem.Evaluation; 
Problem.FEs = ProblemNew.FEs;

%% Initialization 
if nargin == 5
    initPop1 =  max(Problem.lower', min(initPop, Problem.upper'));
    [Population,FEs] = LMOEAInitialization(Problem,t,CostFunction,FEs,initPop1);
    Pareto.init_PopPOF = (Population.objs)';
    Pareto.init_Pop = (Population.decs)';
    POF_iter.init_PopPOF = (Population.objs)';
    POF_iter.init_Pop = (Population.decs)';
    Population = DominationSelection(Problem,Population); 
else
    [Population,FEs] = LMOEAInitialization(Problem,t,CostFunction,FEs);
    Pareto.init_Pop = Population.decs';
    Pareto.init_PopPOF = (Population.objs)';
    POF_iter.init_PopPOF = (Population.objs)';
    POF_iter.init_Pop = (Population.decs)';
end
[RefV,~]   = UniformPoint(5*N,M);  %100*2 double

while FEs<maxFEs
    [GuidingSolution,FEs] = DirectedSampling(Problem,Population,Ns,Nw,RefV,FEs,t);
    [Population,FEs] = DoubleReproduction(Problem,Population,GuidingSolution,RefV,FEs,t);
end

Population1 = DominationSelection(Problem,Population);
Population = Population1;
Pareto.X = (Population.decs)';
Pareto.F = (Population.objs)';
PopX = Pareto.X;
POF_iter.X = (Population1.decs)';
POF_iter.F = (Population1.objs)';
end






