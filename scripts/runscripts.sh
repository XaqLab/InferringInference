echo "Submitting simulations"
BRAIN_NAME="Ns_25_noiseseed_4021"
MODEL_TYPE="1"
NOISE_SEED="0"
B="2000"
NP="100"
ITERS="100"
BSIZE="3"
LR="4e-3"
QOBS="0.02"

for ns_hat in {0..31}
do
	taskset --cpu-list $ns_hat python InferTAPbrain.py $BRAIN_NAME $MODEL_TYPE $NOISE_SEED $ns_hat $B $NP $ITERS $BSIZE $LR $QOBS > OutLogs/Ns_hat_$ns_hat-Trial_$NOISE_SEED.txt &
	sleep 2
done