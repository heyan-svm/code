function [w1,w2,bias1,bias2] = svc(c,d,C1,p,z1,z2)
% Parameters of the L2,p-GEPSVM
% c: positive samples
% d: negative samples
% C1: a regularization factor
% p: p-order
% z1 and z2 are the solutions of standard GEPSVM

[m,n]=size(c);e=ones(m,1);
[m2,n2]=size(d);e2=ones(m2,1);

H=[c -e];
G=[d -e2];

it1=0;
delta=1e+50;
while(delta>0.00001 && it1<50)  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
temp1=H*z1;
temp_store1=(sqrt(temp1(:,1).^2)).^(p-2);
D11=sparse(diag(temp_store1));
U1=H'*D11*H+C1*speye(n+1);

temp2=G*z1;
temp_store2=(sqrt(temp2(:,1).^2)).^(p-2);
D12=sparse(diag(temp_store2));
V1=G'*D12*G;

[AA,BB]=eig(U1,V1);
B1=diag(BB);
[B1,index11]=min(B1);
z1_new=AA(:,index11(1,1));
% the objective function value
Obj_new1=(norm(G*z1_new,2).^p)\(norm(H*z1_new,2).^p);
 if it1>1
        delta=Obj_old1-Obj_new1;
 end

z1=z1_new;
Obj_old1=Obj_new1;
it1=it1+1;
end
w1=z1(1:end-1,1);
bias1=z1(end,1);


it2=0;
delta=1e+50;
while(delta>0.00001 && it2<50)  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
temp3=G*z2;
temp_store3=(sqrt(temp3(:,1).^2)).^(p-2);
D21=sparse(diag(temp_store3));
U2=G'*D21*G+C1*speye(n2+1);

temp4=H*z2;
temp_store4=(sqrt(temp4(:,1).^2)).^(p-2);
D22=sparse(diag(temp_store4));
V2=H'*D22*H;

[AA2,BB2]=eig(U2,V2);
B2=diag(BB2);
[B2,index22]=min(B2);
z2_new=AA2(:,index22(1,1));
% the objective function value
Obj_new2=abs((norm(H*z2_new,2).^p)\(norm(G*z2_new,2).^p));

 if it2>1
        delta=Obj_old2-Obj_new2;
 end
 
z2=z2_new;
Obj_old2=Obj_new2;
it2=it2+1;
end 
w2=z2(1:end-1,1);
bias2=z2(end,1);

end
    