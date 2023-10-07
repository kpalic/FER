#!/bin/bash

# $a='2'.5.'2'; 
# $b='python $a'; 
# $c="Guido "van' Rossum'; 
# for i in $a $b $c; 
# do 
#    echo -n "$i "; 
# done

# a='2'.5.'2'; b='python $a'; c="Guido "van' Rossum'; for i in $a $b $c; do echo -n "$i "; done


# k="cygutil12 cygutil14 aaa libgcc12 libgcc14 libint21 libint24.lst termcap.dsc cygutil12.lst cygutil14.dsc libgcc12.lst libgcc14.dsc libint22.lst libint42 termcap.lst"
# echo "$k" | grep -o '[^t]*12'

# j=3
# echo "gaj"
# i=2
# echo "gaj"
# z=$((j+i))
# echo "gaj"
# bash
# echo "gaj"
# j=1
# echo "gaj"
# bash
# echo $i
# exit
# echo $j
# exit
# echo $z

# for i in {1..4}; do j=0; s=""; while [ $j -lt $i ]; do s="$s+j"; j=$(($j+1)); done; echo $s; done

# j=3; 
# z=5;
# [ $j -lt 4 ] && [ $z -ge 6 ] && j=$(($j+2)); 
# [ $j -gt 4 ] || j=$(($j+3)); 
# echo $j;

# a="nula devet devet devet sest pet jedan tri sedam osam"
# echo $a | sed -r 's/\b[^ ]{3}/3/g' | sed -r 's/\b3[^ ]/4/g' | sed -r 's/\b4[^ ]/5/g' 
# | sed -r 's/\b5[^ ]/6/g'

# for i in 6 3 5; do j=”$i+2”; k=”$i+$j”; echo -n $k; done > d; echo $(($(cat d)))

# a="Tcl/Tk 8.5.8
# Ruby 1.9.1
# Python 2.6.5
# Python 3.1.2
# Perl 5.10.1"
# # echo $a | sed -r 's/[^[:alpha:] ]//g' | sed -r 's/[ ][ ]*/\n/g'

j=3
i=2
z=$((j+i))
bash
j=1
echo $i
exit
echo $j
exit
echo $z