CONFIG=$1 
CKPT=$2
GPUS=$3

echo ${CONFIG}
echo ${CKPT}
echo ${GPUS}

bash ./tools/dist_test.sh \
    ${CONFIG} \
    ${CKPT} \
    ${GPUS} \
    --deterministic \
    --eval bbox

