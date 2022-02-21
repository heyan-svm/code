function[w1,w2,u1,u2,bias11,bias12,bias21,bias22 ]=IMVGEPSVM(A1,A2,B1,B2,det,epis1,epis2)

[m1,n1]=size(A1);e=ones(m1,1);
MA1=[A1 e];
G1= [A1 e]'*[A1 e];
[m2,n]=size(A2);e=ones(m2,1);
MA2=[A2 e];
G2= [A2 e]'*[A2 e];

[m3,n]=size(B1);e=ones(m3,1);
MB1=[B1 e];
H1= [B1 e]'*[B1 e];
[m4,n]=size(B2);e=ones(m4,1);
MB2=[B2 e];
H2= [B2 e]'*[B2 e];

K1_1=blkdiag(G1,G2);
K1_2=fliplr(blkdiag(fliplr(MA1'*MA2),fliplr(MA2'*MA1)));
K1=(1+det)*K1_1+(-det)*K1_2;
K1=K1+epis1*eye(size(K1));

T1=blkdiag(H1,H2);
[eigVector,eigValue]=eig(K1,T1);
eigValue=diag(eigValue);
[eigValue,index1]=min(eigValue);
Z=eigVector(:,index1(1,1));
%--------------------------------------------------------------------------
winit1=Z(1:size(A1,2),1);%First view
bias11=Z(size(A1,2)+1,1);
winit2=Z(size(A1,2)+2:end-1,1);
bias21=Z(end,1);
%--------------------------------------------------------------------------
K2_1=blkdiag(H1,H2);
K2_2=fliplr(blkdiag(fliplr(MB1'*MB2),fliplr(MB2'*MB1)));
K2=(1+det)*K2_1+(-det)*K2_2;
K2=K2+epis1*eye(size(K2));

T2=blkdiag(G1,G2);  
[eigVector2,eigValue2]=eig(K2,T2);
eigValue2=diag(eigValue2);
[eigValue2,index2]=min(eigValue2);
P=eigVector2(:,index2(1,1));
%--------------------------------------------------------------------------
uinit1=P(1:size(A1,2),1);%Second view
bias12=P(size(A1,2)+1,1);
uinit2=P(size(A1,2)+2:end-1,1);
bias22=P(end,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
value1=inf;
value2=inf;
max_t=51;
epsion=0.001;
v=1e-8;
t1=1;
t2=1;
w1=winit1;w2=winit2;
u1=uinit1;u2=uinit2;

[m1,n1]=size(A1);
e11=ones(m1,1);
[m1,n2]=size(A2);
e21=ones(m1,1);
[m2,n1]=size(B1);
e12=ones(m2,1);
[m2,n2]=size(B2);
e22=ones(m2,1);

A1=[A1 e11];
w1=[w1;bias11];
A2=[A2 e21]; 
w2=[w2;bias21];
w=[w1;w2];
B1=[B1 e12];
u1=[u1;bias12];
B2=[B2 e22];
u2=[u2;bias22];
u=[u1;u2];

objList1=[];
objList11=[];
objList2=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MA1=blkdiag(A1,A2);
MA3=[A1,-A2];

MB1=blkdiag(B1,B2);
MB3=[B1,-B2];

E1=eye(size(w1,1));
E2=eye(size(w2,1));
E=blkdiag(E1,E2);
[HHH,QQQ]=VC(A1,A2,B1,B2);
while ( value1>epsion && t1<=max_t)
w1=w(1:size(w1,1),1);
w2=w(size(w1,1)+1:end,1);
lambda_t1=(norm(A1*w1,1)+norm(A2*w2,1)+det*norm(w1,2)+det*norm(w2,2)+epis1*norm((A1*w1-A2*w2),1))/(norm(B1*w1,1)+norm(B2*w2,1)+epis2*norm(HHH'*w,1));

hi=1./(sqrt((MA1*w+v).^2));
D1=diag(hi);
ki=1./(sqrt((MA3*w+v).^2));
K=diag(ki);

sb1=sign(w1'*B1')*B1;
sb2=sign(w2'*B2')*B2;
sb=[sb1,sb2];
sh=sign(w'*HHH)*HHH';
AAA=(MA1'*D1*MA1+epis1*MA3'*K*MA3+det*E)+(MA1'*D1*MA1+epis1*MA3'*K*MA3+det*E)';

BBB=lambda_t1*(sb+sh*epis2)';
wnew=AAA\BBB;
wnew1=wnew(1:size(w1,1),1);
wnew2=wnew(size(w1,1)+1:end,1);
if(t1>1)
value1old=(norm(A1*w1,1)+norm(A2*w2,1)+det*norm(w1,2)+det*norm(w2,2)+epis1*norm((A1*w1-A2*w2),1))/(norm(B1*w1,1)+norm(B2*w2,1)+epis2*norm(HHH'*w,1));
value1new=(norm(A1*wnew1,1)+norm(A2*wnew2,1)+det*norm(wnew1,2)+det*norm(wnew2,2)+epis1*norm((A1*wnew1-A2*wnew2),1))/(norm(B1*wnew1,1)+norm(B2*wnew2,1)+epis2*norm(HHH'*wnew,1));

value1=abs(value1old-value1new);
objList1=[objList1;value1new];
objList11=[objList11;value1];
w=wnew;
end
t1=t1+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (value2>epsion && t2<=max_t)
u1=u(1:size(u1,1),1);
u2=u(size(u1,1)+1:end,1);

lambda_t2=(norm(B1*u1,1)+norm(B2*u2,1)+det*norm(u1,2)+det*norm(u2,2)+epis1*norm((B1*u1-B2*u2),1))/(norm(A1*u1,1)+norm(A2*u2,1)+epis2*norm(QQQ'*u,1));

hi=1./(sqrt((MB1*u).^2));
H=diag(hi);

ki=1./(sqrt((MB3*u).^2));
K=diag(ki);

sc1=sign(u1'*A1')*A1;
sc2=sign(u2'*A2')*A2;
sc=[sc1,sc2];
sq=sign(u'*QQQ)*QQQ';
 
AAA2=(MB1'*H*MB1+epis1*MB3'*K*MB3+det*E)+(MB1'*H*MB1+epis1*MB3'*K*MB3+det*E)';
BBB2=lambda_t2*(sc+sq*epis2)';
unew=AAA2\BBB2;
unew1=unew(1:size(u1,1),1);
unew2=unew(size(u1,1)+1:end,1);
if(t2>1)

value2old=(norm(B1*u1,1)+norm(B2*u2,1)+det*norm(u1,2)+det*norm(u2,2)+epis1*norm((B1*u1-B2*u2),1))/(norm(A1*u1,1)+norm(A2*u2,1)+epis2*norm(QQQ'*u,1));
value2new=(norm(B1*unew1,1)+norm(B2*unew2,1)+det*norm(unew1,2)+det*norm(unew2,2)+epis1*norm((B1*unew1-B2*unew2),1))/(norm(A1*unew1,1)+norm(A2*unew2,1)+epis2*norm(QQQ'*unew,1));
value2=abs(value2old-value2new);
objList2=[objList2;value2new];
u=unew;
end
t2=t2+1;
end