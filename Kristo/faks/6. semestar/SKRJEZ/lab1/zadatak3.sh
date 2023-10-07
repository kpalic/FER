#!/bin/bash

# Za svaku datoteku treba zapisati datum, a zatim za svaku akciju koja se pojavljuje u
# logovima treba ispisati koliko se puta dogodila toga dana.

# Podatke o akcijama sortirati prema silaznom
# broju ponavljanja, broj ponavljanja ispisati prije same akcije.

# datum je u formatu dd-mm-gggg

# Skripta kao argument naredbenog retka prima ime direktorija u kojemu se nalaze log-datoteke

# Skripta treba provjeriti postoji li zadani direktorij te u sluˇcaju pogreške
# ispisati uputu o naˇcinu korištenja i iza ́ci iz skripte.

# datum="$(awk '{ print $4 }' ./logs/localhost_access_log.2008-02-24.txt | cut -f1 -d":" | uniq | cut -f2 -d "[" | tr "/" -)"
# echo "Datum : $datum"
# echo "--------------------"
# awk '{ print $6 $7 $8 }' ./logs/localhost_access_log.2008-02-24.txt | cut -f1 -d"?" | tr "\"" : | sort | uniq -c | sort -r

#name=${file##*/}
search_dir=$1
for entry in "$search_dir"/*
do
    echo "$entry"
    datum="$(awk '{ print $4 }' "$entry" | cut -f1 -d":" | uniq | cut -f2 -d "[" | tr "/" -)"
    echo "Datum : $datum"
    echo "--------------------------------------------------------"
    awk '{ print $6 $7 $8 }' "$entry" | cut -f1 -d"?" | cut -f2 -d"[" | sort | uniq -c | sort -r | awk '{print $1 ": " $2}'
done
