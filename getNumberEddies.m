function [alpha, beta, SI] = getNumberEddies(RytovVar)

RRV = sqrt(RytovVar);

alpha = (exp((0.49*RRV^2)/((1 + 1.11*RRV^(12/5))^(7/6))) -1)^-1;
beta  = (exp((0.51*RRV^2)/((1 + 0.69*RRV^(12/5))^(5/6))) -1)^-1;

SI = exp((0.49*RRV^2)/((1 + 1.11*RRV^(12/5))^(7/6)) ...
    + (0.51*RRV^2)/((1 + 0.69*RRV^(12/5))^(5/6))) - 1;