#!/bin/sh
echo "Submitting simulations"
for i in {0..2}
do
  python InferTAPbrain.py Ns_5_noiseseed_20 42 150 30 $i > "OutLogs/log_$i.txt" &
done
