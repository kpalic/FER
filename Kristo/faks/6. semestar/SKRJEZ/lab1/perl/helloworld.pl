#!/usr/bin/perl -w

print "Hello svijete\n";
print 5 / 3 . "\n";
print 5.5 / 3.2 . "\n";

$what = "brontosaurus steak";
$n = 3;
print "fred ate $n $whats.\n"; # varijabla je $whats
print "fred ate $n ${what}s.\n"; # sad je ime $what
print "fred ate $n $what" . "s.\n"; # moze i ovako

$line = <STDIN>;
if ($line eq "\n") {
    print "To je samo prazni redak!\n";
}
else {
    print "Ucitani redak je: $line";
}

$text = $line; # ili niz ucitan sa <STDIN>
print "okej " . $text;
print "okej " . $text;
print "okej " . $text;
chomp($text);
print $text;
print $text;
print $text;
