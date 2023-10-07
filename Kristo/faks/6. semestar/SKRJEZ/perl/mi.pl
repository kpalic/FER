#!/bin/perl

# $a='2'.5.'2'; 
# $b='python $a'; 
# $c="Guido "van' Rossum'; 
# for i in $a $b $c; 
# do 
#    echo -n "$i "; 
# done

#a='2'.5.'2'; b='python $a'; c="Guido "van' Rossum'; for i in $a $b $c; do echo -n "$i "; done

# $a="2.5"; $b=4; $c="4.0"; $d=2.5; $e=2;
# if($a eq $q || $a == $e){print "jedan ";}
# if($a == $d && $a == $e){print "dva ";}
# if($a ne $d || $b == $c){print "tri ";}
# if($a eq $d && $b eq $c){print "cetiri ";}
# if($a eq $d){print "ad ";}
# if($a == $d){print "ad ";}
# if($b eq $c){print "bc ";}
# if($b == $c){print "bc ";}

# $k="cygutil12 cygutil14 libgcc12 libgcc14 libint21 libint24.lst termcap.dsc cygutil12.lst cygutil14.dsc libgcc12.lst libgcc14.dsc libint22.lst libint42 termcap.lst";
# echo $k;

@a = (4,3,2,1); @b = (5, @a, 6, 7, 69); $c = @b; print $c . ":" . $b[1];