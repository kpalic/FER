#!/bin/bash

# Napisati naredbu grep koja  ́ce u datoteci namirnice.txt prona ́ci i ispisati sve retke u kojima se
# pojavljuju nazivi vo ́ca (banana, jabuka, jagoda, dinja, lubenica), neovisno o tome jesu li
# napisani velikim ili malim slovima

grep -n -i "banana\|jabuka\|jagoda\|dinja\|lubenica" namirnice.txt
echo "--------------"

# modificirati prethodnu naredbu tako da se ispisuju samo retci u kojima se ne pojavljuju zadane rijeˇci.
grep -i "banana\|jabuka\|jagoda\|dinja\|lubenica" namirnice.txt
echo "--------------"

# Napisati naredbu grep koja ce u kazalu ~/projekti/ i svim njegovim podkazalima pronaci datoteke
# u kojima se pojavljuje šifra u obliku tri velika slova i šesteroznamenkasti broj, te ispisati retke u
# kojima se ta šifra pojavljuje. Šifra od ostalog teksta mora biti odvojena razmakom

grep -r -P '[[:blank:]][[:upper:]]{3}[[:digit:]]{6}[[:blank:]]' ~/projekti
echo "--------------"

# samo stvar koju trazimo :
echo "Ili ako zelimo samo sifru bez ostalog teksta : "
grep -r -o -P '[[:blank:]][[:upper:]]{3}[[:digit:]]{6}[[:blank:]]' ~/projekti
echo "--------------"

# Napisati naredbu koja  ́ce ispisati imena i detaljne podatke svih datoteka u teku ́cem kazalu i njegovim
# podkazalima, koje su mijenjane prije 7 do 14 dana.

# . pokazuje na trenutno kazalo, naredbi je potreban samo pocetni direktorij, pretrazivanje je rekurzivno
# -14 je pocetno vrijeme da nemamo +7 pokazivalo bi nam datume mijenjane izmedu -14 dana i danas
find . -type f -mtime -14 -mtime +7 -ls
echo "--------------"

# Napisati u jednom retku for petlju koja  ́ce ispisati brojeve od 1 do 15. Pritom iskoristiti izraz za
# generiranje sekvence ili naredbu seq
for i in $(seq 1 15); do echo $i; done
echo "--------------"

# Modificirati prethodnu naredbu tako da se zadnji broj raspona zadaje u varijabli kraj. Provjeriti je
# li svejedno koristi li se izraz za generiranje sekvence ili naredba seq.
kraj=15
for i in {1..kraj}; do echo $i; done
echo "--------------"
for i in $(seq 1 $kraj); do echo $i; done
echo "--------------"
# postoji razlika prilikom koristenja varijable kraj


