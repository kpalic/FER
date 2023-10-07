#!/usr/bin/perl

use strict;
use warnings;

my $filename = shift @ARGV;
open(my $fh, '<', $filename) or die "Cannot open file '$filename': $!";

while (my $line = <$fh>) {
    chomp $line;
    my ($jmbag, $prezime, $ime, $termin, $zakljucano) = split /;/, $line;

    # Izdvajanje datuma i vremena početka termina
    # my ($datum, $pocetak) = ($termin =~ /(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})/);
    my ($datum, $pocetak) = split / /, $termin, 4;

    # print "datum : $datum\npocetak : $pocetak\n";

    # Izdvajanje datuma i vremena kada je test zaključan
    my ($datum_zakljucano, $vrijeme_zakljucano) = split / /, $zakljucano, 2;

    # Provjera jesu li početak termina i vrijeme zaključavanja u istom satu
    my ($sat_pocetak) = ($pocetak =~ /^(\d{2})/);
    my ($sat_zakljucano) = split /:/, $vrijeme_zakljucano, 2;
    if ($sat_pocetak == $sat_zakljucano) {
        next;  # Student je zaključao test unutar prvog sata, preskačemo
    }

    # Inače, student nije zaključao test unutar prvog sata, ispisujemo problem
    printf("%s %s %s - PROBLEM: %s %s --> %s\n", $jmbag, $prezime, $ime, $datum, $pocetak, $vrijeme_zakljucano);
}

close($fh);
