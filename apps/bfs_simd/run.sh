#!/usr/bin/bash
make clean
make
set -x

# Tile SIze
data_file="/storage/datafolder/0.9b14b.txt"
#output="output_$(date +%Y%m%d-%H%M%S)_livejournal.txt"
#:> $output
#for ((i = 2048; i < 65537; i *= 2)); do
#	./bfs ${data_file} ${i} 16 &>> $output
#done
./bfs ${data_file} 8192 16 16
# Twt
#data_file="/data/zpeng/twt_combine/tiled_2-power/out.twitter"
#output="output_$(date +%Y%m%d-%H%M%S)_twt.txt"
#:> $output
#for ((i = 16384; i < 524289; i *= 2)); do
#	./bfs ${data_file} ${i} 40 &>> $output
#done
set +x
