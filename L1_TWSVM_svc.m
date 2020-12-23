function [w1,w2,b1,b2] = svc(c,d,C1,C2)
% c: positive samples
% d: negative samples
% C1,C2: the regularization factors

[m,n]=size(c);e1=ones(m,1);
[m2,n2]=size(d);e2=ones(m2,1);
H=[c e1];
G=[d e2];

dd=H'*H+1e-7*speye(n2+1);
H1=G*(dd\G');
H1=(H1+H1')/2;

alpha=qpSOR(H1,0.7,C1,0.05);
u=-(dd\G')*alpha;

it=0;
delta=1e+50;

while(delta>0.001 && it<50)  
wold=u;
wt=abs(H*wold); 
D1=diag(1./(sqrt(wt.^2+1e-14)));
alpha=qpSOR(H1,0.7,C1,0.05);

    t=H'*D1*H;
    temp1=t +1e-7*speye(n2+1);
    %%%%%%%%%%%%%%%%%%%
    % to avoid the singularity
    [x1,x2]=size(temp1);

    jk=7.1;
    while( x2~=rank(temp1) && jk<15)
    temp1=t +10^jk*speye(n2+1);
    [x1,x2]=size(temp1);
    if x2==rank(temp1)
    break;
    end
    jk=jk+1;
    end
%%%%%%%%%%%%%%%%%%%%%%%
    u=-(temp1\G')*alpha;
%%%%%%%%%%%%%%%%%%%%%%%
q1=e2+G*u;
q=max(0,q1);
obnew=norm(H*u,1)+C1*e2'*q;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if it>1
        delta=obold-obnew;
    end
    obold=obnew;
    it=it+1;
 w1=u(1:n,1);
 b1=u(n+1,1);
end


 dd2=G'*G+1e-7*speye(n+1);
  H2=H*(dd2\H'); 
  H2=(H2+H2')/2;
alpha2=qpSOR(H2,0.7,C2,0.05);
v=(dd2\H')*alpha2;

it=0;
delta=1e+50;
while(delta>0.001 && it<50)
wold=v; 
wt=abs(G*wold);

D2=diag(1./(sqrt(wt.^2+1e-14)));
alpha2=qpSOR(H2,0.7,C2,0.05);
    t=G'*D2*G;
    temp2=t +1e-7*speye(n+1);
    %%%%%%%%%%%%%%
    [x1,x2]=size(temp2);
    jk=7.1;
    while (x2~=rank(temp2) && jk<15)
    temp2=t +10^jk*speye(n+1);
    [x1,x2]=size(temp2);
        if x2==rank(temp2)
        break;
          end
    jk=jk+1;
    end
    %%%%%%%%%%%%%
    v=(temp2\H')*alpha2; 
%%%%%%%%%%%%%%%%%%%%%%%%
q2=e1-H*v;
q=max(0,q2);
obnew=norm(G*v,1)+C2*e1'*q;
%%%%%%%%%%%%%%%%%%%%%%%%%%
if it>1
    delta=obold-obnew;
end 
 obold=obnew;
 it=it+1;
 w2=v(1:n,1);
 b2=v(n+1,1);
end
