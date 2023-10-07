#!/bin/bash

# Argumenti: navodi se ime datoteke u kojoj se nalaze podaci
arg=$1

# Pripaziti na ispravno pozivanje skripte te napisati informativnu poruku u slučaju krivog pozivanja!
if [ $# -ne 1 ]
  then
    echo "Krivo koristenje skripte, pogresan broj argumenata"
    exit 1
fi

# korisnicko ime, lozinka, id, group_id, ime' 'prezime,  
# poredati po prezime -> count -> descending

cat $arg | cut -f5 -d':' | cut -f2 -d" " | sort -r | uniq -c | sort -r



