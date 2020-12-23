function [w1,w2,bias1,bias2] = svc(c,d,C)
% c: positive samples
% d: negative samples
% C: a regularization factor

[m,n]=size(c);e=ones(m,1);
[m2,n2]=size(d);e2=ones(m2,1);

H=[d -e2]'*[d -e2];
M=[c -e]'*[c -e];

G=M+C*speye(n+1);
L=H+C*speye(n2+1);

[A,B]=eig(G,H);
B=diag(B);

[B,index1]=min(B);

[A2,B2]=eig(L,M);

B2=diag(B2);
[B2,index2]=min(B2);

vector=A(:,index1(1,1));
vector2=A2(:,index2(1,1));

w1=vector(1:end-1,1);
w2=vector2(1:end-1,1);

bias1=vector(end,1);
bias2=vector2(end,1);
