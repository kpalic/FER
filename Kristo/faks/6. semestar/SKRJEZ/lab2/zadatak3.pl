#!/usr/bin/perl
foreach my $filename (@ARGV) {
    open(my $fh, '<', $filename) or die "Cannot open file '$filename': $!";
    while (my $line = <$fh>) {
        if ($line =~ /([a-zA-Z]*)(href="http)([s]?)(.*)pdf/) {
            my ($text1, $text2, $text1) = split /"/, $line, 3;
            print "$text2\n";
        }
    }
}