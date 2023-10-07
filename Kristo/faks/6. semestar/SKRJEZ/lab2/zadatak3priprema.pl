#!/usr/bin/perl

my %counter;

foreach my $filename (@ARGV) {
    open(my $fh, '<', $filename) or die "Cannot open file '$filename': $!";
    while (my $line = <$fh>) {
        chomp $line;
        my ($ip, $date_time, $action) = parse_log_line($line);
        $date_time = substr($date_time, 1);
        my ($date, $time) = split /:/, $date_time, 2; 
        my ($hour) = ($time =~ /^(\d\d)/);
        $counter{$date}{$hour}++;
    }

    close($fh);
}

if (keys %counter == 0) {
    while (my $line = <>) {
        chomp $line;
        my ($ip, $date_time, $action) = parse_log_line($line);
        my ($date, $time) = split /\s+/, $date_time;
        my ($hour) = $time =~ /^(\d\d)/;
        $counter{$date}{$hour}++;
    }
}

foreach my $date (sort keys %counter) {
    print "Datum: $date\n";
    print "sat : broj pristupa\n";
    print "-------------------------------\n";
    foreach my $hour (0..23) {
        my $hour_str = sprintf("%02d", $hour);
        my $count = $counter{$date}{$hour_str} // 0;
        print "$hour_str : $count\n";
    }
}

sub parse_log_line {
    my ($line) = @_;
    my ($ip, $dash1, $dash2, $date_time, $zone, $request, $status, $bytes) = split /\s+/, $line;
    return ($ip, $date_time, $request);
}
