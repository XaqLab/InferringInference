echo "Submitting simulations"
ITERS="10"
NSHAT="5"
QOBS="0.05"
for i in {0..15}
do
	taskset --cpu-list $i python Loglikelihood-vs-Nshat.py $i $ITERS $NSHAT $QOBS > OutLogs/log-Nshat$NSHAT-trial$i.txt &
done