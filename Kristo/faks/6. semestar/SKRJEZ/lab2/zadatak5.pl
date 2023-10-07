#!/usr/bin/perl

use strict;
use warnings;

# učitavanje putanje datoteke kao argumenta komandne linije
my $file_path = shift;

# provjera postojanja datoteke
if (not -e $file_path) {
    die "Datoteka ne postoji!\n";
}

# otvaranje datoteke za čitanje
open my $fh, '<', $file_path or die "Neuspjelo otvaranje datoteke: $!";

# inicijalizacija potrebnih varijabli
my @factors;
my %results;

# čitanje podataka iz datoteke
while (my $line = <$fh>) {

    # preskoči komentirane i prazne retke
    next if $line =~ /^\s*#/ or $line =~ /^\s*$/;

    # učitavanje faktora
    if (not @factors) {
        @factors = split ';', $line;
        next;
    }

    # učitavanje rezultata
    my ($id, $name, $surname, @marks) = split ';', $line;
    print "id: $id\nname: $name\nsurname: $surname\n";
    print "marks: ";
    foreach my $a (@marks) {
        print "$a\n";
    }
    # preskoči retke u kojima nedostaje id ili prezime i ime
    next if not defined $id or not defined $name;

    # računanje ukupnog broja bodova za studenta
    my $total_marks = 0;
    foreach my $i (0 .. $#marks) {
        next if $marks[$i] eq '-';
        $total_marks += $marks[$i] * $factors[$i];
    }

    # spremanje rezultata za studenta
    $results{$id} = [$name, $total_marks];
}

# zatvaranje datoteke
close $fh;

# sortiranje i ispis rang-liste
my $rank = 1;
print "Lista po rangu:\n-------------------\n";
foreach my $id (sort { $results{$b}[1] <=> $results{$a}[1] } keys %results) {
    printf "%d. %s (%s) : %.2f\n", $rank++, $results{$id}[0], $id, $results{$id}[1];
}
