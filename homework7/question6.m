function [R,quad,err,h]=romber(f,a,b,n,tol)
    %Input - f is the integrand input as a string ’f’
    %- a and b are upper and lower limits of integration
    %- n is the maximum number of rows in the table
    %- tol is the tolerance
    %Output - R is the Romberg table
    %- quad is the quadrature value
    %- err is the error estimate
    %- h is the smallest step size used
    M=1;
    h=b-a;
    err=1;
    J=0;
    R=zeros(4,4);
    R(1,1)=h*(feval(f,a)+feval(f,b))/2;
    while((err>tol)&(J<n))|(J<4)
        J=J+1;
        h=h/2;
        s=0;
        for p=1:M
            x=a+h*(2*p-1);
            s=s+feval(f,x);
        end
        R(J+1,1)=R(J,1)/2+h*s;
        M=2*M;
        for K=1:J
            R(J+1,K+1)=R(J+1,K)+(R(J+1,K)-R(J,K))/(4^K-1);
        end
        err=abs(R(J,J)-R(J+1,K+1));
    end
    quad=R(J+1,J+1);
end


function y=f1(x)
    y = sqrt(4*x-x*x);
end

function y=f2(x)
    y=4/(1+x*x);
end

for i = 1:23
    [R1, quad1, err1, h1]=romber(@f1, 0, 2, i, 1e-10);
    [R2, quad2, err2, h2]=romber(@f2, 0, 1, i, 1e-10);
    fprintf('n=%d,  quad1=%d,  err1=%d, quad2=%d, err2=%d\n', i, quad1,err1, quad2, err2);
end
