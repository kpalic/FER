#!/bin/bash

# Provjeri da li je korisnik unio direktorij
if [ -z "$1" ]; then
    echo "Molimo unesite putanju do root direktorija."
    exit 1
fi

ROOT_DIR="$1"

# Ako direktorij ne postoji
if [ ! -d "$ROOT_DIR" ]; then
    echo "Direktorij $ROOT_DIR ne postoji."
    exit 1
fi

# Ako ispis.txt već postoji, obriši ga
if [ -e "ispis.txt" ]; then
    rm "ispis.txt"
fi

# Prolazi kroz sve datoteke u direktoriju i svim njegovim poddirektorijima
find "$ROOT_DIR" -type f | while read -r file; do
    echo "Putanja do datoteke: $file" >> ispis.txt
    echo "Sadržaj datoteke:" >> ispis.txt
    # Dodaj iconv -f UTF-8 da bi se osiguralo da je ispis u UTF-8 formatu
    iconv -f UTF-8 "$file" >> ispis.txt
    echo "" >> ispis.txt
done

echo "Završeno. Provjerite datoteku ispis.txt za rezultate."
