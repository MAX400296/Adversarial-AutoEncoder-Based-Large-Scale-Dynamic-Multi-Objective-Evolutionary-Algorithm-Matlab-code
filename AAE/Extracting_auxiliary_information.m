% Developed in MATLAB R2021b
% Source codes demo version 1.0
% _____________________________________________________
% Main paper:
% Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm
% IEEE Transactions on Evolutionary Computation
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [change_direction, dltat_1tot,historyPos,historyPof,Local_change_distance_max,Local_change_distance_min,dlta,cos_dlta] = Extracting_auxiliary_information(historyPos,historyPof,dlta,cos_dlta)
%%%%%%%%%%%%%%%%  historical POSs are matched to each other by the nearest Euclidean distance of their corresponding POFs  %%%%%%%%%%%%%%%%%%%%%%%%
k = size(historyPof,3);
Y = historyPof(:,:,k)';
X = historyPof(:,:,k-1)';
Dis = pdist2(Y,X);
%%%%%%%%%%%%%Call Munkres  Assignment Algorithm %%%%%%%%%%%%%%%%%%%%%%
%address: https://brc2.com/the-algorithm-workshop/
[assignment,~] = munkres(Dis);
[assignedrows,~]=find(assignment);
assigning_elements = assignedrows;
historyPos(:,:,k) = historyPos(:,assigning_elements,k);
historyPof(:,:,k) = historyPof(:,assigning_elements,k);

%%%%%%%%%%%%%%% Extract change distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%
dlta(:,:,k-1) =historyPos(:,:,k)-historyPos(:,:,k-1);
dltat_1tot = dlta(:,:,end);
distance = abs(dlta);
Local_change_distance = mean(distance,3);  
Local_change_distance_std = std(distance,0,3); 
Local_change_distance_max = Local_change_distance+Local_change_distance_std; 
Local_change_distance_min = Local_change_distance - Local_change_distance_std; 
Local_change_distance_min(Local_change_distance_min<0) = 0;
%change_distance = Local_change_distance;

%%%%%%%%%%%%%%%%%%%%% Angle trends %%%%%%%%%%%%%%%%%%%%%%%%
if nargin == 4
    k1 = size(dlta,3);   
    for i = 1:size(dlta,2)
        dr(i,:) = CalculateCosineSimilarity(dlta(:,i,end)',dlta(:,i,k1-1)');
    end
   cos_dlta(:,:,k1-1) = dr;

    if size(cos_dlta,3) <2
        change_direction = cos_dlta(:,:,end);
        change_direction = cat(2,change_direction,change_direction);
    end
    %Markov chain predictor
    if size(cos_dlta,3) >=2
        data = reshape(cos_dlta,size(cos_dlta,1),size(cos_dlta,3));
        next_observation = Markov_chain_predictor(data);
        change_direction = next_observation;
    end
else
    cos_dlta = [];
    change_direction = [];
end
end

% %%%%%%%%%%%%% Help function %%%%%%
function cosine_similarity = CalculateCosineSimilarity(v1,v2)
dot_product = dot(v1, v2);
% Calculate the magnitudes of the two vectors
magnitude1 = norm(v1);
magnitude2 = norm(v2);
% Calculate the cosine similarity of the two vectors
cosine_similarity = dot_product / ((magnitude1 * magnitude2)+eps);
end




