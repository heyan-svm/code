function [w1,w2,b1,b2] = svc(c,d,lamda)
% c: positive samples
% d: negative samples
% lamda: a regularization factor

[m,n]=size(c);e1=ones(m,1);
[m2,n2]=size(d);e2=ones(m2,1);

H=[d -e2]'*[d -e2];
M=[c -e1]'*[c -e1];

G=M+lamda*speye(n+1);
L=H+lamda*speye(n2+1);

[A,B]=eig(G,H);
B=diag(B);
[B,index1]=min(B);
w0=A(:,index1(1,1));

wnew=w0;

G=[c -e1];
H=[d -e2];

delta_J=1e+50;
max_it=50;
epsilon=0.001;
t=1;

  while (delta_J>epsilon && t<=max_it)        
     
     den1=norm(wnew'*H',1);
     den2=norm(wnew'*G',1);
    Jnew=den1/den2;
     
       if t>3
        delta_J=Jnew-Jold;
      end
    
     Jold=Jnew;
     sp1=H'*sign(wnew'*H')';
     sp2=G'*sign(wnew'*G')'; 
     g=sp1./den1-sp2./den2;    
     wold=wnew;
     wnew=wold+0.0005*g;
 
 wnew=wnew./norm(wnew);
     t=t+1;
 w1=wnew(1:n,1);
 b1=wnew(n+1,1);
  end
  
[A2,B2]=eig(L,M);
B2=diag(B2);
[B2,index2]=min(B2);
v=A2(:,index2(1,1));
     
 wnew1=v;

delta_J=inf;
max_it=50;
epsilon=0.001;
t=1;
  while (delta_J>epsilon && t<=max_it)        
     
     den1=norm(wnew1'*G',1);
     den2=norm(wnew1'*H',1);
     Jnew=den1/den2;
     
       if t>3
        delta_J=Jnew-Jold;
      end

     Jold=Jnew;
     sp1=G'*sign(wnew1'*G')';
     sp2=H'*sign(wnew1'*H')'; 
     g=sp1./den1-sp2./den2;   
     wold=wnew1;
     wnew1=wold+0.0005*g;
 
 wnew1=wnew1./norm(wnew1);
     t=t+1;
 w2=wnew1(1:n,1);
 b2=wnew1(n+1,1);
  end
