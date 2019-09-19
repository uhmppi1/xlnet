#!/usr/bin/env bash

SAVE_DIR='./datagen'
CORPUS_TO_PRETRAIN='./corpus/spm_data/aaa.txt'
MODEL_PREFIX='sp10m.cased.v3'
VOCAB_SIZE=500

cd ..   # BASEDIR인 xlnet 디렉토리에서 실행한다.
BASEDIR=$(pwd)


python train_spm.py \
  --input_spm=${CORPUS_TO_PRETRAIN} \
  --spm_model_prefix=${MODEL_PREFIX} \
  --spm_vocab_size=${VOCAB_SIZE}
