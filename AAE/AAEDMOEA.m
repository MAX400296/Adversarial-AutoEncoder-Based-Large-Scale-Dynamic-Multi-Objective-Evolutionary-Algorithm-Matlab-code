% Developed in MATLAB R2021b
% Source codes demo version 1.0
% _____________________________________________________
% Main paper:
% Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm
% IEEE Transactions on Evolutionary Computation
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INPUT
% Problem: an DMOP；
%popSize：Population size；
%Maxlt：Maximum number of iterations;
%T_parameter: The frequency and severity of changes.
%% OUTPUT
%res: POS, POF.
% To run AAEDMOEA: res=AAEDMOEA(Problem,popSize,MaxIt,T_parameter,group)
function res=AAEDMOEA(Problem,popSize,MaxIt,T_parameter,group)
disp('Run AAEDMOEA')
samplenum = 200;
dlta = [];
cos_dlta = [];
res = cell(1,T_parameter(group,3)/T_parameter(group,2));
for T = 1:T_parameter(group,3)/T_parameter(group,2)
    fprintf(' %d',T);
    t= 1/T_parameter(group,1)*(T);
    Problem.FEs = 0; 
    if T==1
        [~,Pareto,~] = LMOEADS1( Problem,popSize,MaxIt, t);
        s = size(Pareto.F,2);
        [FrontNo,~] = NDSort(Pareto.F',s);
        Next = false(1,length(FrontNo));
        Next(FrontNo<=1) = true;
        EPX = Pareto.X(:,Next);
        EPF = Pareto.F(:,Next);
        PopF=EPF;
        PopX = EPX;
        Post_1 = EPX;
        Act = EPX;
        Dnum=samplenum-size(PopX,2);
        if Dnum>0
            while Dnum >size(PopX,2)
                PopX = cat(2,PopX,PopX);
                Dnum=samplenum-size(PopX,2);
                PopF = cat(2,PopF,PopF);
            end
            randsampling = randperm(size(PopX,2), Dnum) ; 
            PopX = cat(2,PopX,PopX(:,randsampling));
            PopF = cat(2,PopF,PopF(:,randsampling));
        end
        historyPos = PopX;
        historyPof = PopF;
    elseif T<=3
        [~,Pareto,~] = LMOEADS1( Problem,popSize,MaxIt, t,Act);
        s = size(Pareto.F,2);
        [FrontNo,~] = NDSort(Pareto.F',s);
        Next = false(1,length(FrontNo));
        Next(FrontNo<=1) = true;
        EPX = Pareto.X(:,Next);
        EPF = Pareto.F(:,Next);
        Act = cat(2,Act,EPX); 
        Post_1 = EPX;
        PopF=EPF;
        PopX = EPX;
        if size(PopX,2)~=size(historyPos,2)
            Dnum=(size(historyPos,2)-size(PopX,2));
            if Dnum>0
                while Dnum >size(PopX,2)
                    PopX = cat(2,PopX,PopX);
                    Dnum=(size(historyPos,2)-size(PopX,2));
                    PopF = cat(2,PopF,PopF);
                end
                randsampling = randperm(size(PopX,2), Dnum) ; 
                PopX = cat(2,PopX,PopX(:,randsampling));
                PopF = cat(2,PopF,PopF(:,randsampling));
            else
                randsampling = randperm(size(PopX,2), size(historyPos,2)) ; 
                PopX = PopX(:,randsampling);
                PopF = PopF(:,randsampling);
            end
        end
        historyPos = cat(3,historyPos,PopX);
        historyPof = cat(3,historyPof,PopF);
        [~,~,historyPos,historyPof,~,~,dlta,cos_dlta] = Extracting_auxiliary_information(historyPos,historyPof,dlta);
    else
        %%%%%%%%%%%%%%%% AAE-DMOEA %%%%%%%%%%%%%%%%
        [change_direction,dltat_1tot,historyPos,historyPof,Local_change_distance_max,Local_change_distance_min,dlta,cos_dlta] = Extracting_auxiliary_information(historyPos,historyPof,dlta,cos_dlta);
        [Generate_POSt1,~]=AAE(historyPos(:,:,end),change_direction',dltat_1tot,Local_change_distance_max,Local_change_distance_min,Problem,Post_1);

        [~,Pareto,~] = LMOEADS1( Problem,popSize,MaxIt, t, Generate_POSt1);
        s = size(Pareto.F,2);
        [FrontNo,~] = NDSort(Pareto.F',s);
        Next = false(1,length(FrontNo));
        Next(FrontNo<=1) = true;
        EPX = Pareto.X(:,Next);
        EPF = Pareto.F(:,Next);
        Act = cat(2,Act,EPX);
        Post_1 = EPX;
        PopF=EPF;
        PopX = EPX;
        if size(PopX,2)~=size(historyPos,2)
            Dnum=(size(historyPos,2)-size(PopX,2));
            if Dnum>0
                while Dnum >size(PopX,2)
                    PopX = cat(2,PopX,PopX);
                    Dnum=(size(historyPos,2)-size(PopX,2));
                    PopF = cat(2,PopF,PopF);
                end
                randsampling = randperm(size(PopX,2), Dnum) ; 
                PopX = cat(2,PopX,PopX(:,randsampling));
                PopF = cat(2,PopF,PopF(:,randsampling));
            else
                randsampling = randperm(size(PopX,2), size(historyPos,2)) ; 
                PopX = PopX(:,randsampling);
                PopF = PopF(:,randsampling);
            end
        end
        historyPos = cat(3,historyPos,PopX);
        historyPof = cat(3,historyPof,PopF);
    end
    
    res{T}.turePOF=getBenchmarkPOF(Problem.Name,group,T,T_parameter );
    res{T}.POS=Pareto.X;
    res{T}.POF = Pareto.F;
    res{T}.initPop = Pareto.init_Pop;
    res{T}.initPOF = Pareto.init_PopPOF;
end
end

function POF_Banchmark = getBenchmarkPOF( testfunname,group,T,T_parameter )
%UNTITLED2 Summary of this function goes here
tempPosition = T;
% ['./Metrics/pof/measures/pof/' 'POF-nt' num2str(T_parameter(group,1)) '-taut' num2str(T_parameter(group,2)) '-' functions{testfunc} '-' num2str(tempPosition) '.txt']
POF_Banchmark = importdata(['pof/' 'POF-nt' num2str(T_parameter(group,1)) '-taut' num2str(T_parameter(group,2)) '-' testfunname '-' num2str(tempPosition) '.txt']);
end