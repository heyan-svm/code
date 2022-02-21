% function [w1, b1, w2, b2] = RLpNPSVM(Atrain,Btrain,FunPara,w0,itmax,epsmax)
function [w1, b1, w2, b2] = RLpNPSVM(Atrain,Btrain,FunPara)
% Input:
%   Atrain: Positive class data matrix. Each row vector of Atrain is a data point.
%   Btrain: Positive class data matrix. Each row vector of Btrain is a data point.
%   delta: regularization term parameter.
%   FunPara.p: p in Lp-norm distance 
%   FunPara.q: q in Lq-norm regularization term 
%   FunPara.delta: parameter \delta
%   FunPara.sigma: parameter \sigma 
%   w0: Initial hyperplane direction and bias
%   itmax: Maximun iteration term num
%   epsmax: Tolerance
% 
% % % % Usage eample:
%   Atrain = rand(30,2);
%   Btrain = rand(30,2);
%   FunPara.p = 1; 
%   FunPara.q = 2;
%   FunPara.delta = 0.1;
%   FunPara.sigma = 0.1;
%   w0 = ones(size(Atrain,2) + 1,1); % Initialization
%   itmax = 20;
%   epsmax = 10^(-3);
%   [w1, b1, w2, b2] = RLpNPSVM(Atrain,Btrain,FunPara,w0,itmax,epsmax);
% 
% Reference:
%    Generalized elastic net Lp-norm nonparallel support vector machine
%    Chun-Na Li,Pei-Wei Ren, Yuan-Hai Shao, Ya-Fen Ye
%    Version 2.0 -- June/2019
%    Written by Pei-Wei Ren and Chun-Na Li (na1013na@163.com)

Rp = FunPara.p;
Rq = FunPara.q;
Rdelta = FunPara.delta;
Rsigma = FunPara.sigma;
if nargin == 2
    Rp = 2; Rq = 2; Rdelta = 0.01; Rsigma = 0.01; w0 = ones(size(Atrain,2) + 1,1); itmax = 20; epsmax = 10^(-3); 
elseif nargin == 3
    w0 = ones(size(Atrain,2) + 1,1); itmax = 20; epsmax = 10^(-3); 
elseif nargin == 4
    itmax = 20; epsmax = 10^(-3); 
elseif nargin == 5
    epsmax = 10^(-3); 
elseif nargin == 1 || nargin > 6
    sprintf('%s ','Wrong input number')
end

[nSmpA, nFea] = size(Atrain);
[nSmpB, ~] = size(Btrain);
iter = 0;
wp0 = w0;
wn0 = w0;
epsWX = 10^-4;
epsW = 10^-4;
Atrainbar = [Atrain,ones(size(Atrain,1),1)]';
Btrainbar = [Btrain,ones(size(Btrain,1),1)]';

while 1 
    %%%%%%%%%%%%%%%%%%
% % %     For A
    %%%%%%%%%%%%%%%%%%
    iter = iter + 1;
    Ht_A = zeros(nFea+1, nFea+1);
    ht_A = zeros(nFea+1,1);
    for i = 1:nSmpA
        Atrainbari = Atrainbar(:,i);
        Ht_A = Ht_A + (Atrainbari*Atrainbari')/sum((wp0'*Atrainbari + epsWX).^(2 - Rp));
    end
    for i = 1:nSmpB
        ht_A = ht_A + abs(wp0'*Btrainbar(:,i))^(Rp-1)*sign(wp0'*Btrainbar(:,i))*Btrainbar(:,i);
    end
    q_A = (abs(wp0) + epsW).^(Rq - 2+0.0001); 
    Ht_Ar = Rdelta*diag(q_A);
    [s1,~]=size(Ht_Ar);
    Ht_A = Ht_A + Ht_Ar + Rsigma*eye(s1);
    wp = (Ht_A\ht_A)/(ht_A'*inv(Ht_A)*ht_A);
    wp0 = wp;
    %%%%%%%%%%%%%%%%%%
% % %     For B
    %%%%%%%%%%%%%%%%%%
    Ht_B = zeros(nFea+1, nFea+1);
    ht_B = zeros(nFea+1,1);
    for i = 1:nSmpB
        Btrainbari = Btrainbar(:,i);
        Ht_B = Ht_B + (Btrainbari*Btrainbari')/sum((wp0'*Btrainbari + epsWX).^(2 - Rp));
    end
    for i = 1:nSmpA
        ht_B = ht_B + abs(wn0'*Atrainbar(:,i))^(Rp-1)*sign(wn0'*Atrainbar(:,i))*Atrainbar(:,i);
    end
    q_B = (abs(wn0) + epsW).^(Rq - 2); 
    Ht_Br = Rdelta*diag(q_B);
    [s2,~]=size(Ht_Br);
    Ht_B = Ht_B + Ht_Br + Rsigma*eye(s2);
    wn = (Ht_B\ht_B)/(ht_B'*inv(Ht_B)*ht_B);
    wn0 = wn;
    
    if max(abs(wp-wp0))<epsmax && max(abs(wn-wn0))<epsmax || iter>itmax 
        break;
    end
end
w1 = wp(1:length(wp)-1);
b1 = wp(length(wp)); 
w2 = wn(1:length(wn)-1);
b2 = wn(length(wn));
