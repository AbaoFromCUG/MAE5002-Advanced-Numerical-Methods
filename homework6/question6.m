function [L,n]=difflim(f,x,toler)
    %Input - f is the function input as a string ’f’
    %- x is the differentiation point
    %- toler is the tolerance for the error
    %Output-L=[H’ D’ E’]:
    %H is the vector of step sizes
    %D is the vector of approximate derivatives
    %E is the vector of error bounds
    %- n is the coordinate of the ‘‘best approximation’’
    max1=15;
    h=1;
    H(1)=h;
    D(1)=(feval(f,x+h)-feval(f,x-h))/(2*h);
    
    E(1)=0;
    R(1)=0;
    for n=1:2
        h=h/10;
        H(n+1)=h;
        D(n+1)=(feval(f,x+h)-feval(f,x-h))/(2*h);
        E(n+1)=abs(D(n+1)-D(n));
        R(n+1)=2*E(n+1)/(abs(D(n+1))+abs(D(n))+eps);
    end
    n=2;
    while((E(n)>E(n+1))&(R(n)>toler))&n<max1
        h=h/10;
        H(n+2)=h;
        D(n+2)=(feval(f,x+h)-feval(f,x-h))/(2*h);
        E(n+2)=abs(D(n+2)-D(n+1));
        R(n+2)=2*E(n+2)/(abs(D(n+2))+abs(D(n+1))+eps);
        n=n+1;
    end
    n=length(D)-1;
    L=[H' D' E'];
end

function y=f1(x)
    y = vpa(60 * power(x, 45) - 32 * power(x, 33) +233 * power(x, 5) - 47 * power(x, 2) -77);
end

digits(50)
[L1, n1] = difflim(@f1, vpa(1 / sqrt(3)), vpa(1e-13));
format long g;
disp(double(L1));
disp(n1);


function y=f2(x)
    y = vpa(sin(x*x*x - 7*x*x + 6*x + 8));
end

digits(50)
[L2, n2] = difflim(@f2, vpa((1-sqrt(5))/2), vpa(1e-13));
format long g;
disp(double(L2));
disp(n2);
