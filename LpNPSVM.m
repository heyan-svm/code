% function [w1, w2, b1, b2, iter] = LpNPSVM(Atrain,Btrain,delta,p,w0,itmax,epsmax) 
function [w1, w2, b1, b2] = LpNPSVM(Atrain,Btrain,delta,p,w0,itmax,epsmax) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % LpNPSVM: 
% Robust nonparallel proximal support vector machine with Lp-norm regularization
% [w1, b1, w2, b2] = LpNPSVM(Atrain,Btrain,FunPara,w0,itmax,epsmax) 
% Input:
%   Atrain: Positive class data matrix. Each row vector of Atrain is a data point.
%   Btrain: Positive class data matrix. Each row vector of Btrain is a data point.
%   delta: regularization term parameter.
%   FunPara: parameters
%         FunPara.delta: regularization parameter
%         FunPara.p: p in Lp-norm regularization term 
%   w0: Initial hyperplane direction and bias
%   itmax: Maximun iteration number
%   epsmax: Tolerance
% 
% % % % Eample:
% Atrain = rand(30,2);
% Btrain = rand(30,2) + 1;
% w0 = ones(1,size(Atrain,2) + 1); % Initialization
% FunPara.p = 1.5; 
% FunPara.delta = 0.05; 
% [w1, b1, w2, b2] = LpNPSVM(Atrain,Btrain,FunPara);

% Reference:
%    Robust nonparallel proximal support vector machine with Lp-norm regularization. %    IEEE Access, 2018, 6: 20334-20347.
%    Xiao-Quan Sun, Yi-Jian Chen, Yuan-Hai Shao, Chun-Na Li*, Chang-Hui Wang
%    Version 2.0 --June/2018 
%    Written by Chun-Na Li (na1013na@163.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 2
    FunPara.delta = 0.05; FunPara.p = 1.5; w0 = ones(size(Atrain,2) + 1,1); itmax = 100; epsmax = 10^(-5); 
elseif nargin == 3
    w0 = ones(size(Atrain,2) + 1,1); itmax = 100; epsmax = 10^(-5); 
elseif nargin == 4
    itmax = 30; epsmax = 10^(-5); 
elseif nargin == 5
    epsmax = 10^(-5); 
end

w0 = ones(size(Atrain,2) + 1,1);

[nSmpA, nFea] = size(Atrain);
[nSmpB, ~] = size(Btrain);
wp0 = w0;
wn0 = w0;
epsWX = 10^-4;
epsW = 10^-4;
Atrainbar = [Atrain,ones(size(Atrain,1),1)]';
Btrainbar = [Btrain,ones(size(Btrain,1),1)]';
t=1;
obj_delta=10^3;
while (obj_delta>epsmax&&t<itmax)
    %%%%%%%%%%%%%%%%%%
% % %     For A
    %%%%%%%%%%%%%%%%%%
    Ht_A = zeros(nFea+1, nFea+1);
    ht_A = zeros(nFea+1,1);
    for i = 1:nSmpA
        Atrainbari = Atrainbar(:,i);
        Ht_A = Ht_A + (Atrainbari*Atrainbari')/abs(wp0'*Atrainbari + epsWX);
    end
    for i = 1:nSmpB
        ht_A = ht_A + sign(wp0'*Btrainbar(:,i))*Btrainbar(:,i);
    end
    q_A = (abs(wp0) + epsW).^(p - 2); 
    Ht_Ar = delta*diag(q_A);
    Ht_A = Ht_A + Ht_Ar;
    wp = (Ht_A\ht_A)/(ht_A'*inv(Ht_A)*ht_A);
    wp0 = wp;
    %%%%%%%%%%%%%%%%%%
% % %     For B
    %%%%%%%%%%%%%%%%%%
    Ht_B = zeros(nFea+1, nFea+1);
    ht_B = zeros(nFea+1,1);
    for i = 1:nSmpB
        Btrainbari = Btrainbar(:,i);
        Ht_B = Ht_B + (Btrainbari*Btrainbari')/abs(wp0'*Btrainbari + epsWX);
    end
    for i = 1:nSmpA
        ht_B = ht_B + sign(wn0'*Atrainbar(:,i))*Atrainbar(:,i);
    end
    q_B = (abs(wn0) + epsW).^(p - 2); 
    Ht_Br = delta*diag(q_B);
    Ht_B = Ht_B + Ht_Br;
    wn = (Ht_B\ht_B)/(ht_B'*inv(Ht_B)*ht_B);
    obj_old=(norm(wn0,1)+delta*(norm(wn0,p))^p)/(norm(wn0,1));
    obj_new=(norm(wn,1)+delta*(norm(wn,p))^p)/(norm(wn,1));
    obj_delta=abs(obj_old-obj_new);
    wn0 = wn;
    t=t+1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%     if max(abs(wp-wp0))itmax 
%         break;
%     end
end
w1 = wp(1:length(wp)-1);
b1 = wp(length(wp)); 
w2 = wn(1:length(wn)-1);
b2 = wn(length(wn));
