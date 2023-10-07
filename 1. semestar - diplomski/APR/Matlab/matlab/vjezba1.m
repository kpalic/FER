% Rjesava sustav linearnih jednadzbi
% Matrica sustava se ucitava iz matricaA.txt, a slobodni vektor iz vektorb.txt
clear
load matricaA.txt
load vektorb.txt
[L,U,P]=lu(matricaA)
y=L\(P*vektorb) % supstitucija unaprijed
x=U\y           % supstitucija unatrag
disp('Rjesenje dobiveno sa x=A\b:')
matricaA\vektorb
disp('Inverzija matrice sustava:')
inv(matricaA)