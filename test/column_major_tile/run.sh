#!/usr/bin/bash
make clean && make
row_step=8
(set -x; ./kcore ~/benchmarks/data/twt_col/out.twitter 4096 $row_step)
for ((i=5; i<7; ++i)); do
	row_step=$((2**i))
	(set -x; ./kcore ~/benchmarks/data/twt_col/out.twitter 4096 $row_step)
done
echo done.
