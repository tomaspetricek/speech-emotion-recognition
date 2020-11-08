#! /usr/bin/perl	

use File::Basename;
use File::Path qw(make_path remove_tree);

use warnings;
use strict;
	
my $file = $ARGV[0];

my $p;
open F, "<" . $file;
binmode F;
read(F, $p, 12);
my ($nSamples, $sampPeriod, $sampSize, $parmKind) = unpack("LLss", $p);
my $buff;
#my @data = unpack("f*", <F>);

my @data;
while(read F, $buff, 4){
	push(@data, unpack("f", $buff));
}
close F;

open G, "> " . $file . ".txt";

print G $nSamples . "\n";
print G $sampPeriod . "\n";
print G $sampSize . "\n";
print G $parmKind . "\n";

foreach my $dato (@data){
	print G $dato . "\n";
}

