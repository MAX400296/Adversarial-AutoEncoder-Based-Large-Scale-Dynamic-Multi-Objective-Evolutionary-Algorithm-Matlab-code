% Developed in MATLAB R2021b
% Source codes demo version 1.0
% _____________________________________________________
% Main paper:
% Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm
% IEEE Transactions on Evolutionary Computation
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function next_observation = Markov_chain_predictor(data)
%%%%%%%%%%% Parameter setting and discrete partitioning %%%%%%%%%%%%%%%%
n = 10;         
x = linspace(-1, 1, n+1);   
t = (x(1:end-1) + x(2:end))/2;   
for k = 1:size(data,2)
    for k1 = 1:size(data,1)
        value = data(k1,k);   
        idx = find(t<=value, 1, 'last');   
        if isempty(idx)   
            Observe_data(k1,k) = 1;
        else
            time_segment = idx + 1;   
            Observe_data(k1,k) = time_segment;
        end
    end
end
add_zeros = zeros(size(Observe_data,1),1);
data = cat(2,Observe_data,add_zeros);
data=data';
data=data(:)';

%%%%%%%%%%%%% Markov modelling and prediction %%%%%%%%%%%%%%%%
Q =  n+1; 
A = zeros(Q,Q); 
for i=1:Q
    for j=1:Q
            A(i,j)=length(strfind(data,[i,j]));
    end
end
k_sum=sum(A,2);
A1 = (repmat(k_sum,1,size(A,2))+eps);
transition_matrix=A./A1;
next_observation = zeros(size(Observe_data,1),2);
for i = 1:size(Observe_data,1)
    [M,I] = max(transition_matrix(Observe_data(i,end),:));
    if M == 0  
        next_observation_index = Observe_data(i,end);
    else
       next_observation_index = I;
    end
     if next_observation_index==1
        next_observation(i,1) = -1;
        next_observation(i,2) = t(next_observation_index);
    elseif next_observation_index== Q
        next_observation(i,1) = t(next_observation_index-1);
        next_observation(i,2) = 1;
    else
        next_observation(i,1) = t(next_observation_index-1);
        next_observation(i,2) = t(next_observation_index);
     end
end
end