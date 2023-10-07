#!/usr/bin/perl

print "Unesite niz brojeva odvojenih razmakom: ";
my $input = <STDIN>;
chomp($input);

my @brojevi = split(" ", $input);
my $suma = 0;

foreach my $broj (@brojevi) {
    $suma += $broj;
}

my $prosjek = $suma / scalar(@brojevi);
print "Aritmetička sredina je: $prosjek\n";
