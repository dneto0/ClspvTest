#! /usr/bin/env perl

# Parse and emit grid-based comparison from logcat output

use strict;
my $in = 0;
my $dim = 15;
my %expected = ();
my %observed = ();
foreach my $x(0.. $dim) {
    $expected{$x} = {};
    $observed{$x} = {};
}


my $kernel = '';
sub show() {

print "$kernel\nExpected \n";
foreach my $yi(0.. $dim) {
  my $y = $dim - $yi;
foreach my $x(0.. $dim) {
	printf " %2d", $expected{$x}{$y};
}
print "\n";
}
print "\n";
print "\n";

print "Observed \n";
foreach my $yi(0.. $dim) {
  my $y = $dim - $yi;
foreach my $x(0.. $dim) {
	printf " %2d", $observed{$x}{$y};
}
print "\n";
}

}
my $num_values = 0;
my $seen_values = 0;

foreach my $line (<>) {
  if ($in) {
	if ($line =~ m/correctValues:(\d+) incorrectValues:(\d+)/) {
		$num_values = $1 + $2;
		$seen_values = 0;
	}
	if ($line =~ m/x:(\d+), y:(\d+).*expected:. (\S+) . observed:. (\S+) ./) {
		my ($x, $y, $e, $o) = ($1, $2, $3, $4);
		#print "HW ";
		$expected{$x}{$y} = $e + 0;
		$observed{$x}{$y} = $o + 0;
		$seen_values++;
		show() if ($seen_values == $num_values);
	}
	if ($line =~ m/Kernel:/) {
		#	$in = 0;
	}
  }

	if ($line =~ m/Kernel:(\S+)$/) {
		$in = 1;
		$kernel = $1;
	} elsif ($line =~ m/Kernel:(\S+) SKIPPED$/) {
		$in = 0;
		$kernel = '';
	}

}
