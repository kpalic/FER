#!/usr/bin/perl
use strict;
use warnings;
use open ':locale';
use locale;
use utf8;

my ($prefix_len, @files) = @ARGV;
my %prefixes;

# Ako nisu navedene datoteke, čitaj tekst sa standardnog ulaza
if (scalar @files == 0) {
    print "Upišite tekst:\n";
    my $text = join "", <STDIN>;
    count_words($text);
}
# Inače, čitaj datoteke
else {
    foreach my $file (@files) {
        open my $fh, '<:encoding(UTF-8)', $file or die "Nije moguće otvoriti datoteku '$file': $!";
        my $text = join "", <$fh>;
        close $fh;
        count_words($text);
    }
}

# Ispiši riječi sa zajedničkim prefiksom duljine $prefix_len
foreach my $prefix (sort keys %prefixes) {
    my $count = $prefixes{$prefix};
    printf "%s : %d\n", $prefix, $count;
}

sub count_words {
    my ($text) = @_;
    # Ukloni interpunkcijske znakove
    $text =~ s/[^\pL\s]//g;

    # Razbij tekst na riječi i prebaci ih u hash
    foreach my $word (split /\s+/, $text) {
        # Ukloni višak slova
        $word = substr($word, 0, $prefix_len);
        next if length $word < $prefix_len;
        $word = lc $word;  # Pretvori riječ u lowercase
        $prefixes{$word}++;
    }
}
