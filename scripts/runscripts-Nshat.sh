echo "Submitting simulations"
ITERS="5"
QOBS="0.05"
for nshat in {2..9}
do
    for i in {0..15}
    do
        taskset --cpu-list $i python Loglikelihood-vs-Nshat.py $i $ITERS $nshat $QOBS > OutLogs/log-Nshat$nshat-trial$i.txt &
    done
    wait
done