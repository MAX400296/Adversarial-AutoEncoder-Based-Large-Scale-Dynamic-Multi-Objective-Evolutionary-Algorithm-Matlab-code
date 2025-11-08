function [Population,FrontNo,CrowdDis] = DominationSelection(Global,Population)
% The dominant relationship and crowding based environmental selection

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Shufen Qin
% E-mail: shufen.qin@stu.tyust.edu.cn

    %% Non-dominated sorting
    %Global.N = size(Population,2);
    [FrontNo,MaxFNo] = NDSort(Population.objs,inf);
    %[FrontNo,MaxFNo] = NDSort(Population.objs,Global.N);
    Next = false(1,length(FrontNo));
    %Next(FrontNo<2) = true;
    %Next(FrontNo<MaxFNo) = true;
    Next(FrontNo==1) = true;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    A = sum(Next);
    %% Select the solutions in the last front based on their crowding distances  
    if sum(Next)<Global.N
        %Last     = find(FrontNo==MaxFNo);
        Last     = find(FrontNo>=2);
        [~,Rank] = sort(CrowdDis(Last),'descend');
        if Global.N>size(Population,2)
            Next(Last(Rank(1:size(Population,2)-sum(Next)))) = true;
        else
            Next(Last(Rank(1:Global.N-sum(Next)))) = true;
        end
    else
        Last     = find(FrontNo==1);
        %Last     = find(FrontNo>=2);
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next1(Last(Rank(1:Global.N))) = true;
        Next = Next1;
    end
    
    
    %% Population for next generation
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
    Population = Population(Next);
end

function CrowdDis = CrowdingDistance(PopObj,FrontNo)
%CrowdingDistance - Calculate the crowding distances of solutions front
%by front.
%
%   CD = CrowdingDistance(F) calculates the crowding distances of solutions
%   according to their objective values in F.
%
%   CD = CrowdingDistance(F,FrontNo) calculates the crowding distances of
%   solutions in each non-dominated front, where FrontNo is the front
%   numbers of solutions.
%
%   Example:
%       CrowdDis = CrowdingDistance(PopObj,FrontNo)

%------------------------------- Reference --------------------------------
% S. Kukkonen and K. Deb, Improved pruning of non-dominated solutions based
% on crowding distance for bi-objective optimization problems, Proceedings
% of the IEEE Congress on Evolutionary Computation, 2006, 1179-1186.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    [N,M] = size(PopObj);
    if nargin < 2
        FrontNo = ones(1,N);
    end
    CrowdDis = zeros(1,N);
    Fronts   = setdiff(unique(FrontNo),inf);
    for f = 1 : length(Fronts)
        Front = find(FrontNo==Fronts(f));
        Fmax  = max(PopObj(Front,:),[],1);
        Fmin  = min(PopObj(Front,:),[],1);
        for i = 1 : M
            [~,Rank] = sortrows(PopObj(Front,i));
            CrowdDis(Front(Rank(1)))   = inf;
            CrowdDis(Front(Rank(end))) = inf;
            for j = 2 : length(Front)-1
                CrowdDis(Front(Rank(j))) = CrowdDis(Front(Rank(j)))+(PopObj(Front(Rank(j+1)),i)-PopObj(Front(Rank(j-1)),i))/(Fmax(i)-Fmin(i));
            end
        end
    end
end