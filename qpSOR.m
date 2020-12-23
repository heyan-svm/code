function bestalpha=qpSOR(Q,t,C,smallvalue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Q: input matrix£»
%    t: the parameter, belongs to£¨0£¬2£©£»
%    C:the up bound£»
%    smallvalue:terminal condition£»
%    bestalpha:output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization parameter 
[m,n]=size(Q);
alpha0=zeros(m,1);
L=tril(Q);
E=diag(Q);
twinalpha=alpha0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:n
    twinalpha(j,1)=alpha0(j,1)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1)-1+L(j,:)*(twinalpha(:,1)-alpha0));
    if twinalpha(j,1)<0
        twinalpha(j,1)=0;
    elseif twinalpha(j,1)>C
        twinalpha(j,1)=C;
    else
        ;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=[alpha0,twinalpha];
ite=0;
while norm(alpha(:,2)-alpha(:,1))>smallvalue && ite<100
    ite=ite+1;
    for j=1:n
        twinalpha(j,1)=alpha(j,2)-(t/E(j,1))*(Q(j,:)*twinalpha(:,1)-1+L(j,:)*(twinalpha(:,1)-alpha(:,2)));
        if twinalpha(j,1)<0
            twinalpha(j,1)=0;
        elseif twinalpha(j,1)>C
            twinalpha(j,1)=C;
        else
            ;
        end
    end
    alpha(:,1)=[];
    alpha=[alpha,twinalpha];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output
bestalpha=alpha(:,2);
