function y = sinc(x)            %mình tự thêm vào
%SINC Normalized sinc function: sin(pi*x)/(pi*x)
% Works without toolboxes.
y = ones(size(x));
idx = (x ~= 0);
y(idx) = sin(pi*x(idx)) ./ (pi*x(idx));
end