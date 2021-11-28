CUDA_VISIBLE_DEVICES="0,1"
MTPATH=/home/ruanjunhao/translate
code=/home/ruanjunhao/translate/CS
dataset=${MTPATH}/iwslt14.tokenized.de-en
nohup python3 ${code}/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts \
        --world_size 2 \
        --gpus 2 \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 2048  > ${MTPATH}/mt.ckpts/log.txt  &

