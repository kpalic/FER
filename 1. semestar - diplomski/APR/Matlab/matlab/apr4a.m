% crtanje f4 = |(x - y) * (x + y)| + (x^2 + y^2)^0.5
clear;
 x = -1:0.05:1.0;
 y  = -1:0.05:1.0;

 z=(x')*y;% za inicijalizaciju polja

 for i = 1:41,
    for j = 1:41,
        z(i,j) = abs((x(i)-y(j))*(x(i)+y(j)))+sqrt(x(i)*x(i)+y(j)*y(j));
    end
end
surf(x,y,z);