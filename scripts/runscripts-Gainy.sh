echo "Submitting simulations"
ITERS="5"
QOBS="0.05"
NSHAT="5"

for gainy in {1..25}
do
    for i in {0..15}
    do
        taskset --cpu-list $i python Loglikelihood-vs-Gainy.py $i $ITERS $NSHAT $QOBS $gainy > OutLogs/log-Gainy$gainy-trial$i.txt &
    done
    wait
done