% Developed in MATLAB R2021b
% Source codes demo version 1.0
% _____________________________________________________
% Main paper:
% Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm
% IEEE Transactions on Evolutionary Computation
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
close all

Function_name_all = {'DF1'}; 
num_str = cell(1, 100);  
for i = 1:100
    num_str{i} = sprintf('T%d', i);
end
algorithmName={'AAEDMOEA'};
algrithmNum=size(algorithmName,2); 
T_parameters = [
    10 5 100
    10 10 200
    10 25 500
    10 50 1000
    1 10 200
    1 50 1000
    20 10 200
    5 10 300];
T_parameter=T_parameters(8,:); 
popSize=100; 
group = 1;
MaxIt = 5000*T_parameter(2); 
function_num = 1; 
start_function = 1; 
dimention = 100; 

% % % RUN %%%%%%%%%%%%%%%%%%%%%%%%%
for kp = start_function:function_num
    disp("Test Function DF"+kp);
    Problem=TestFunctions(Function_name_all{kp},dimention); 
    CostFunction=Problem.FObj;  % Cost Function
    for cnum=1:algrithmNum 
        alg_fhd=str2func(algorithmName{cnum});
        reskt=alg_fhd(Problem,popSize,MaxIt,T_parameter,group); 
        all_result{kp,cnum} = reskt;
    end
end

%figure
Y = all_result{1,1}{1,30}.POF;
s = size(Y,2);
[FrontNo,~] = NDSort(Y',s);
Next = false(1,length(FrontNo));
Next(FrontNo<=1) = true;
EPF = Y(:,Next);
True_Y = all_result{1,1}{1,30}.turePOF;
figure;
scatter(True_Y(:,1),True_Y(:,2),'b');
hold on
scatter(EPF(1,:),EPF(2,:),'r');
hold off
xlabel('f1');
ylabel('f2');
title('The 30th optimization result.');
legend({'TruePOF','Optimized POF'}, 'Location', 'northeast');



