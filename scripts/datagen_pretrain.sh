#!/usr/bin/env bash

SAVE_DIR='./datagen'
CORPUS_TO_PRETRAIN='./corpus/pretrain/*.txt'

cd ..   # BASEDIR인 xlnet 디렉토리에서 실행한다.
BASEDIR=$(pwd)


python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=16 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=${CORPUS_TO_PRETRAIN} \
	--save_dir=${SAVE_DIR} \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=spiece.model \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85