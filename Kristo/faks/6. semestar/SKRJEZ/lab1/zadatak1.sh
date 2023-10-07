#!/bin/bash

# Postaviti varijablu ljuske proba na vrijednost "Ovo je proba".
proba="Ovo je proba"
# Ispisati vrijednost postavljene varijable.
echo $proba
echo "-------------"
# U varijablu lista_datoteka upisati listu svih datoteka tekuceg kazala. Pritom iskoristiti širenje imena datoteke
lista_datoteka="$(ls -l)"
# Ispisati rezultat
echo "$lista_datoteka" # ako nema dvostrukih navodnika -e ne radi
echo "-------------"


# U varijablu ljuske proba3 upisati 3 puta nadovezanu 
# vrijednost varijable proba, pri cemu na svaku
# recenicu treba dodati tocku i razmak.
proba3=$proba
for i in {1..3}; do
    proba3=$proba3". "$proba
done
echo $proba3
echo "-------------"



# Varijablu a postaviti na vrijednost 4, varijablu b na 3, varijablu c na 7. Zatim u varijablu d upisati
# vrijednost koja se dobije izrazom (a + 4) ∗ b%c. Upotrijebiti širenje aritmetiˇckih izraza. Ispisati
# vrijednosti varijabli a, b, c i d.
a=4
b=3
c=7
d="$((($a+4)*$b%$c))"
echo "a = $a, b = $b, c = $c, d = $d"
echo "-------------"

# U varijablu broj_rijeci upisati ukupan broj rijeci u .txt datotekama teku ́ceg kazala. Upotrijebiti
# supstituciju naredbe i naredbu wc.
broj_rijeci=$(wc -w ./*.txt)
echo "$broj_rijeci"
echo "-------------"

echo ~eaprlik
echo "-------------"


