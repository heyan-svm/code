function [w1,w2,b1,b2] = pTWSVM(c,d,C1,C2,p)

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
D1=norm(wt,2).^(p-2);
alpha=qpSOR(H1,0.7,C1,0.05);
    t1=H'*D1*H;
    temp1=t1 +1e-7*speye(n2+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u=-(temp1\G')*alpha;
q1=e2+G*u;
q=max(0,q1);
obnew=norm(H*u,2).^p+C1*e2'*q;
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

it2=0;
delta2=1e+50;
while(delta2>0.001 && it2<30)
wold=v;
wd=abs(G*wold);
D2=norm(wd,2).^(p-2);
alpha2=qpSOR(H2,0.7,C2,0.05);
    t2=G'*D2*G;
    temp2=t2 +1e-7*speye(n+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v=(temp2\H')*alpha2; 
q2=e1-H*v;
q=max(0,q2);
obnew2=norm(G*v,2).^p+C2*e1'*q;
%%%%%%%%%%%%%%%%%%%%%%%%%%
if it2>1
    delta2=obold2-obnew2;
end 
 obold2=obnew2;
 it2=it2+1;
 w2=v(1:n,1);
 b2=v(n+1,1);
end
