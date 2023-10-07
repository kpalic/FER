#!/bin/bash

base=$1 

find $base -regex "2020[0-9]{2}*"
months=$(ls ./fourth/Testovi/Slike/ | cut -c 1-6 | uniq)

while read -r datum
do
    mjesec=${datum: -2}
    godina=${datum:0:4}

    echo
    echo "-----------"
    echo "$godina-$mjesec :"
    j=0

    naredba="$(find $base -name "*$godina$mjesec*" | cut -f5 -d'/')"
    while read -r line; do
        j=$((j+1))
        echo "$j : $line"
    done <<< "$naredba"
    echo "--- Ukupno: $j slika -----"


done <<< "$months"