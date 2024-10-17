#!/bin/bash

if [ -z "$1" ]; then
    echo "No argument provided. Options: bc2gm, bc5cdr-chem, bc5cdr-disease, jnlpba."
    exit 1
fi

option=$1

model=BioLinkBERT-large
model_path=michiyasunaga/$model


case "$option" in

############################### NER: BC2GM ###############################
    "bc2gm")

        task=BC2GM_hf
        datadir=../data/tokcls/$task
        outdir=runs/$task/$MODEL
        mkdir -p $outdir

        python3 -u run_ner.py \
        --model_name_or_path $model_path \
        --train_file $datadir/train.json \
        --validation_file $datadir/dev.json \
        --test_file $datadir/test.json \
        --do_train \
        --do_eval \
        --do_predict \
        --fp16 \
        --seed 42 \
        --max_seq_length 512 \
        \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-7 \
        --warmup_ratio 0.5 \
        --num_train_epochs 4 \
        --weight_decay 0.01 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --load_best_model_at_end \
        --output_dir $outdir \
        --overwrite_output_dir \
        --log_level warning \
        |& tee $outdir/log.txt

        ;;

############################### NER: BC5CDR-chem ###############################
    "bc5cdr-chem")

        ;;

############################### NER: BC5CDR-disease ###############################
    "bc5cdr-disease")

        ;;

############################### NER: JNLPBA ###############################
    "jnlpba")

        ;;

############################### END ###############################
    *)
        echo "Unknown option: $option"
        exit 1
        ;;
esac