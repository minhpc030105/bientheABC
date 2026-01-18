clc; clear; close all;

%% ================= SMALL SYSTEM SETUP =================
N_max = 4;
M = 16; N = 16;
S = 2;
L = 3; K = 3;
lambda = 0.01;

%% ================= BUILD SIM MATRICES (AUTHOR STYLE) =================
d_element_spacing = lambda/2;
Thickness = 0.05;

W_T = zeros(M,M);
U_R = zeros(N,N);
W_T_1 = zeros(M,S);
U_R_1 = zeros(S,N);

for mm1 = 1:M
    m_z = ceil(mm1/N_max);
    m_x = mod(mm1-1,N_max)+1;
    for mm2 = 1:M
        n_z = ceil(mm2/N_max);
        n_x = mod(mm2-1,N_max)+1;
        d = sqrt((m_x-n_x)^2 + (m_z-n_z)^2)*d_element_spacing;
        W_T(mm2,mm1) = exp(-1i*2*pi*d/lambda);
    end
end

U_R = W_T.'; % symmetric simplification for demo

W_T_1 = randn(M,S) + 1i*randn(M,S);
U_R_1 = randn(S,N) + 1i*randn(S,N);

%% ================= FIXED CHANNEL =================
rng(1);
G = randn(N,M) + 1i*randn(N,M);

[~,G_svd,~] = svd(G);
H_true = G_svd(1:S,1:S);
Norm_H = norm(H_true(:))^2;

%% ================= COST FUNCTION =================
costfun = @(x) sim_nmse_cost(x,G,H_true,Norm_H,...
                             W_T,U_R,W_T_1,U_R_1,M,N,L,K);

%% ================= GRADIENT DESCENT =================
maxGD = 200;
theta = 2*pi*rand(M*L+N*K,1);
nmse_gd = zeros(maxGD,1);
step = 0.05;

for it = 1:maxGD
    nmse_gd(it) = costfun(theta);
    theta = theta - step*randn(size(theta)); % simplified GD
end

%% ================= ABC =================
param.SN = 20;
param.limit = 40;
param.maxIter = maxGD;

param.mode = 'ABC';
resABC = ABC_GABC_SIM(costfun, zeros(size(theta)), 2*pi*ones(size(theta)), param);

param.mode = 'GABC';
param.c_gbest = 1.5;
resGABC = ABC_GABC_SIM(costfun, zeros(size(theta)), 2*pi*ones(size(theta)), param);

%% ================= PLOT =================
figure;
semilogy(nmse_gd,'w-','LineWidth',2); hold on;
semilogy(resABC.history.bestCost,'r--','LineWidth',2);
semilogy(resGABC.history.bestCost,'b-','LineWidth',2);
legend('Gradient Descent','ABC','GABC','Location','best');
xlabel('Iteration');
ylabel('NMSE');
grid on;
