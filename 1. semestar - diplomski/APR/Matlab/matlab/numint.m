% Numericka integracija preko ode funkcija
function numint

tspan=[0 2];
load x0.txt;

opcije=odeset('InitialStep',0.01);

[t,y]=ode45(@sustav, tspan, x0,opcije);
plot(t,y);
title('Runge-Kutta');
pause;

[t1,y1]=ode23t(@sustav, tspan, x0,opcije);
plot(t1,y1);
title('Trapez');

% MORA BITI NA KRAJU DATOTEKE!!!
function s=sustav(t,x)
s=[0*x(1)   + x(2)
  -200*x(1)-1*x(2) ];
%s= 66.66666*x+6.666666;
%s = -0.1*x;
