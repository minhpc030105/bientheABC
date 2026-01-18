function cost = sim_nmse_cost(x, G, H_true, Norm_H, W_T, U_R, W_T_1, U_R_1, M, N, L, K)
% x: phase vector in [0, 2pi], length = M*L + N*K

    % Decode phase vector
    theta_tx = reshape(x(1:M*L), [M, L]);
    theta_rx = reshape(x(M*L+1:end), [N, K]);

    phase_transmit = exp(1i * theta_tx);   % M x L
    phase_receive  = exp(1i * theta_rx);   % N x K

    % ----- TX-SIM response P -----
    P = diag(phase_transmit(:,1)) * W_T_1;
    for l = 1:L-1
        P = diag(phase_transmit(:,l+1)) * W_T * P;
    end

    % ----- RX-SIM response Q -----
    Q = U_R_1 * diag(phase_receive(:,1));
    for k = 1:K-1
        Q = Q * U_R * diag(phase_receive(:,k+1));
    end

    % ----- Effective SIM channel -----
    H_SIM = Q * G * P;

    % ----- NMSE -----
    H_SIM_vec  = H_SIM(:);
    H_true_vec = H_true(:);

    alpha = (H_SIM_vec' * H_SIM_vec) \ (H_SIM_vec' * H_true_vec);
    cost  = norm(alpha * H_SIM_vec - H_true_vec)^2 / Norm_H;
end
