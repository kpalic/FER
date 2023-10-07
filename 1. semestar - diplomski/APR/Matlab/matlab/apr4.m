function y = apr4(x)
% Implementira funckiju za optimiranje
y =  abs((x(1)-x(2))*(x(1)+x(2))) + (x(1)^2+x(2)^2)^0.5

% crtanje iste
clear;
 x = -2:0.1:2.0;
 y  = -2:0.1:2.0;

 z=(x')*y;% za inicijalizaciju polja?

 for i = 1:41,
    for j = 1:41,
        z(i,j) = abs((x(i)-y(j))*(x(i)+y(j)))+sqrt(x(i)*x(i)+y(j)*y(j));
    end
end
surf(x,y,z);