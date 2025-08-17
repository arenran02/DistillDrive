CONFIG=$1 
GPUS=$2 

echo ${CONFIG}
echo ${GPUS}

bash ./tools/dist_train.sh \
   ${CONFIG} \
   ${GPUS} \
   --deterministic