% Developed in MATLAB R2021b
% Source codes demo version 1.0
% _____________________________________________________
% Main paper:
% Adversarial AutoEncoder-based Large-Scale Dynamic Multi-Objective Evolutionary Algorithm
% IEEE Transactions on Evolutionary Computation
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Adversarial AutoEncoder
function [Generate_POS,Loss_value] = AAE(X,change_direction,dltat_1tot,Local_change_distance_max,Local_change_distance_min,Problem,Post_1)

trainX = X;
D = pdist2(Post_1',X');
[~,I]=min(D,[],2);

%% Settings
settings.latent_dim = min(round(size(X,1)/10),5); settings.batch_size = round(size(X,2)); settings.image_size = size(X,1); 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5; settings.beta2 = 0.999;  
settings.maxepochs = 100; %epoch
%% Initialization
%if 1%isempty(model)
%% Encoder
paramsEn.FCW1 = dlarray(initializeGaussian([1024,...
    prod(settings.image_size)],.02));
paramsEn.FCb1 = dlarray(zeros(1024,1,'single'));
%paramsEn.FCW2 = dlarray(initializeGaussian([512,512]));
%paramsEn.FCb2 = dlarray(zeros(512,1,'single'));
paramsEn.FCW3 = dlarray(initializeGaussian([2*settings.latent_dim,1024]));
paramsEn.FCb3 = dlarray(zeros(2*settings.latent_dim,1,'single'));
%% Decoder
paramsDe.FCW1 = dlarray(initializeGaussian([1024,settings.latent_dim],.02));
paramsDe.FCb1 = dlarray(zeros(1024,1,'single'));
%paramsDe.FCW2 = dlarray(initializeGaussian([512,512]));
%paramsDe.FCb2 = dlarray(zeros(512,1,'single'));
paramsDe.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),1024]));
paramsDe.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
%% Discriminator
paramsDis.FCW1 = dlarray(initializeGaussian([1024,settings.latent_dim],.02));
paramsDis.FCb1 = dlarray(zeros(1024,1,'single'));
%paramsDis.FCW2 = dlarray(initializeGaussian([256,512]));
%paramsDis.FCb2 = dlarray(zeros(256,1,'single'));
paramsDis.FCW3 = dlarray(initializeGaussian([1,1024]));
paramsDis.FCb3 = dlarray(zeros(1,1,'single'));

% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.En = []; avgGS.En = [];
avgG.De = []; avgGS.De = [];

%%%%%%%%%%%%%%% Upload  DATA TO GPU%%%%%%%%%%%%
XBatch=gpdl(double(trainX),'CB');
Post_1GPU = gpdl(double(Post_1),'CB');
dltat_1totBatch=gpdl(double(dltat_1tot),'CB');
change_directionBatch = gpdl(double(change_direction),'CB');
Local_change_distance_max = gpdl(double(Local_change_distance_max),'CB');
Local_change_distance_min = gpdl(double(Local_change_distance_min),'CB');

%% Train
out = false; epoch = 0; global_iter = 0; 
Loss_value = [];
sign_out = 0; 
while ~out
    global_iter = global_iter+1;
    [GradEn,GradDe,GradDis,G_loss] = ...
        dlfeval(@modelGradients,XBatch,dltat_1totBatch, change_directionBatch,...
        paramsEn,paramsDe,paramsDis,settings);
 
    % Update Discriminator network parameters
    [paramsDis,avgG.Dis,avgGS.Dis] = ...
        adamupdate(paramsDis, GradDis, ...
        avgG.Dis, avgGS.Dis, global_iter, ...
        settings.lrD, settings.beta1, settings.beta2);

    % Update Encoder network parameters
    [paramsEn,avgG.En,avgGS.En] = ...
        adamupdate(paramsEn, GradEn, ...
        avgG.En, avgGS.En, global_iter, ...
        settings.lrG, settings.beta1, settings.beta2);

    % Update Decoder network parameters
    [paramsDe,avgG.De,avgGS.De] = ...
        adamupdate(paramsDe, GradDe, ...
        avgG.De, avgGS.De, global_iter, ...
        settings.lrG, settings.beta1, settings.beta2);
    epoch = epoch+1;
    if (G_loss(1,1))<=0.005
        sign_out = sign_out+1;
    else
        sign_out = 0;
    end
    if epoch == settings.maxepochs || sign_out>=1
        disp("The total number of AAE runs"+epoch);
        out = true;
    end
end
%Generating initial population
Generate_POS = Generating_initial_population(paramsDe,settings,paramsEn,Post_1GPU,Local_change_distance_max,Local_change_distance_min,Problem,I);
end


%% %%%%%%%%%%%%%%%%%%%%%%% Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%
%% model Gradients
function [GradEn,GradDe,GradDis,G_loss]=modelGradients(x,dltat_1tot, change_direction,paramsEn,paramsDe,paramsDis,settings)
dly = Encoder(x,paramsEn);
latent_fake = dly(1:settings.latent_dim,:)+...
    dly(settings.latent_dim+1:2*settings.latent_dim,:).*...
    randn(settings.latent_dim,settings.batch_size);
latent_real = gpdl(randn(settings.latent_dim,settings.batch_size),'CB');
% Train the discriminator(LOSS FUNCTION)
d_output_fake = Discriminator(latent_fake,paramsDis);
d_output_real = Discriminator(latent_real,paramsDis);
d_loss = -.5*mean(log(d_output_real+eps)+log(1-d_output_fake+eps));
fake_imagesSoure = x; 
fake_imagesTarge = Decoder(latent_fake,paramsDe); 
dltaPt_1_to_Pt = dltat_1tot;
dltaFake_imagesSoure_to_fake_imagesTarge= fake_imagesTarge-fake_imagesSoure;
dr = (sum(dltaPt_1_to_Pt.*(dltaFake_imagesSoure_to_fake_imagesTarge)))./((sqrt(sum(dltaPt_1_to_Pt.^2)+eps)).*(sqrt(sum((dltaFake_imagesSoure_to_fake_imagesTarge).^2)+eps)));
lower_distanse = abs(dr-change_direction(1,:));
upper_distance = abs(change_direction(2,:)-dr);
direction_distance = lower_distanse+upper_distance;
change_direction_distance = abs(change_direction(2,:)-change_direction(1,:));
loss1 = (mean((direction_distance - change_direction_distance).^2));  
g_loss =0.999*(loss1)-0.001*(mean(log(d_output_fake+eps)));
% For each network, calculate the gradients with respect to the loss %%%%%%%
[GradEn,GradDe] = dlgradient(g_loss,paramsEn,paramsDe,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
%INPUT LOSS VALUE %%%%%%%%%
G_loss(1,1) = gatext(g_loss);
end

%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = gpuArray(dlarray(x,labels));
end
%% Weight initialization
function parameter = initializeGaussian(parameterSize,sigma)
if nargin < 2
    sigma = 0.02;
end
parameter = randn(parameterSize, 'single') .* sigma;
end

%% Encoder
function dly = Encoder(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly);
%dly = leakyrelu(dly,.2);
%dly = fullyconnect(dly,params.FCW2,params.FCb2);
%dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly);
%dly = leakyrelu(dly,.2);
end
%% Decoder
function dly = Decoder(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly);
%dly = leakyrelu(dly,.2);
%dly = fullyconnect(dly,params.FCW2,params.FCb2);
%dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
%dly = leakyrelu(dly,.2);
%dly = tanh(dly);
%dly = sigmoid(dly);
end
%% Discriminator
function dly = Discriminator(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
%dly = leakyrelu(dly,.2);
dly = leakyrelu(dly);
%dly = fullyconnect(dly,params.FCW2,params.FCb2);
%dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = sigmoid(dly);
end

%% simpling
function all_sampling_area = Generating_initial_population(paramsDe,settings,paramsEn,x,Local_change_distance_max,Local_change_distance_min,Problem,I)
all_sampling_area = [];
XLow = Problem.XLow;
XUpp = Problem.XUpp;
dly = Encoder(x,paramsEn);
for k = 1:5
    noise = dly(1:settings.latent_dim,:)+...
    dly(settings.latent_dim+1:2*settings.latent_dim,:).*...
    randn(settings.latent_dim,size(x,2));
    Xt1 = Decoder(noise,paramsDe);
    %normalized into 𝑼 which in turn preserves only the direction information.
    globle_chang_direction = Xt1-x;
    Unit_direction = globle_chang_direction;
    for i = 1:size(globle_chang_direction,2)
        S=sum(globle_chang_direction(:,i).*globle_chang_direction(:,i));
        S1 = sqrt(S);
        S2 = globle_chang_direction(:,i)/S1;
        Unit_direction(:,i) = S2;     
    end
    W = Local_change_distance_min(:,I)+(Local_change_distance_max(:,I)-Local_change_distance_min(:,I)).*rand(1,size(Local_change_distance_max(:,I),2));
    X_new = x+Unit_direction.*W;
    sampling_area =X_new;
    sampling_area = max(XLow, min(sampling_area, XUpp));
    sampling_area  = gatext(sampling_area); 
    sampling_area = double(sampling_area); 
    all_sampling_area = cat(2,all_sampling_area,sampling_area);
end
end

