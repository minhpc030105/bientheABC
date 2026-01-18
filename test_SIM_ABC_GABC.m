clc; clear; close all;

% ================= SYSTEM SETUP (REDUCED SIZE) =================
lambda = 0.01;
N_max = 4;
M = 16; N = 16;
S = 2;
L = 3; K = 3;

% Dummy channel matrices for demo (use real ones from repo if ready)
W_T = randn(M,M) + 1i*randn(M,M);
U_R = randn(N,N) + 1i*randn(N,N);
W_T_1 = randn(M,S) + 1i*randn(M,S);
U_R_1 = randn(S,N) + 1i*randn(S,N);
G = randn(N,M) + 1i*randn(N,M);

[~,G_svd,~] = svd(G);
H_true = G_svd(1:S,1:S);
Norm_H = norm(H_true(:))^2;

% ================= ABC PARAMETERS =================
D = M*L + N*K;
lb = zeros(D,1);
ub = 2*pi*ones(D,1);

costfun = @(x) sim_nmse_cost(x,G,H_true,Norm_H,W_T,U_R,W_T_1,U_R_1,M,N,L,K);

param.SN = 20;
param.limit = 40;
param.maxIter = 500;

% ================= RUN ABC =================
param.mode = 'ABC';
resABC = ABC_GABC_SIM(costfun, lb, ub, param);

% ================= RUN GABC =================
param.mode = 'GABC';
param.c_gbest = 1.5;
resGABC = ABC_GABC_SIM(costfun, lb, ub, param);

% ================= PLOT =================
figure;
semilogy(resABC.history.bestCost,'r--','LineWidth',2); hold on;
semilogy(resGABC.history.bestCost,'b-','LineWidth',2);
legend('ABC','GABC');
xlabel('Iteration');
ylabel('NMSE');
grid on;
