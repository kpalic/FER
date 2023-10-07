#!/bin/bash

dirOne=$1
dirTwo=$2

# filesOne="$(ls -l $1)"
# filesTwo="$(ls -l $2)"
# echo "$filesOne"
# echo 
# echo "$filesTwo"
# echo 

for file1 in "$dirOne"/*
do
    exists=$dirTwo${file1: -4}
    if test -f "$exists"
    then
        if test "$file1" -nt "$exists"
        then
            echo "$file1 ---> $2"
        else
            echo "$exists ---> $1"
        fi
    else 
        echo "$file1 ---> $2"
    fi
done

for file2 in "$dirTwo"/*
do
    exists=$dirOne${file2: -4}
    if ! test -f "$exists"
    then
        echo "$file2 ---> $1"
    fi
done