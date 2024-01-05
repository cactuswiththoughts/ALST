#!/bin/bash

tgt_dir=$1
layers=$2
am=$3
pooling=$4
label=$5
if [ $label = vieira ]; then 
    tgt_dir=$tgt_dir/vieira
else
    tgt_dir=$tgt_dir/with_$label
fi
stage=5
stop_stage=6

if [ ! -d $tgt_dir ]; then 
    mkdir -p $tgt_dir 
fi

echo prepare_als.sh stage 0: extract metadata for ALS
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    if [ $label = vieira ]; then
        label_dir=$ALS_WAV_DIR/../Metadata
        python scripts/prepare_vieira_data.py \
            --in-dir $tgt_dir/../ --out-dir $tgt_dir \
            --wav-dir $ALS_WAV_DIR --label-dir $label_dir
    else
        python scripts/prepare_data.py \
            $ALS_WAV_DIR $label_dir $tgt_dir \
            --with-labels $label
        python scripts/convert_kaldi_to_fairseq.py \
            --in-dir $tgt_dir \
            --out-dir $tgt_dir --test-speaker-ratio 0.2
    fi
fi

echo prepare_als.sh stage 1: extract speech features
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for x in train test; do 
        am_prefix=$(echo $am | cut -d '_' -f 1)
        am_suffix=$(echo $am | cut -d '_' -f 2)
        echo $layers
        if [ $am_prefix = 'w2v2' ]; then
            python scripts/wav2vec_extract_features.py \
                $tgt_dir --split $x --save-dir $tgt_dir/$am/feat --checkpoint $W2V2 --layers $layers
        elif [ $am_prefix = 'whisper' ]; then
            python scripts/extract_whisper_feats.py \
                $tgt_dir --split $x --save-dir $tgt_dir/$am/feat --model-name $am_suffix --layers $layers
        fi
    done
fi

echo prepare_als.sh stage 2: extract longitudinal speech features
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for x in train test; do
        for layer in $(echo $layers | sed 's/,/ /g'); do
            python scripts/create_longitudinal_sequence.py \
                $tgt_dir/$am/feat/layer$layer \
                --split $x \
                --save-dir $tgt_dir/$am/feat_$pooling/layer$layer \
                --pooling $pooling
        done
    done
fi

echo prepare_als.sh stage 3: extract phoneme-level forced alignment
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    start_layer=$(echo $layers | cut -d ',' -f 1)
    for x in train test; do
        python scripts/extract_mfa_alignment.py \
            --manifest $tgt_dir/$am/feat/layer$start_layer \
            --align_dir $ALIGN_DIR \
            --out_dir $tgt_dir/$am/feat/layer$start_layer \
            --split $x
    done    

    for layer in $(echo $layers | sed 's/,/ /g'); do
        cp $tgt_dir/$am/feat/layer$start_layer/*.seg $tgt_dir/$am/feat/layer$layer
        cp $tgt_dir/$am/feat/layer$start_layer/*.phn $tgt_dir/$am/feat/layer$layer
    done 

#    vocab_file=$(dirname $W2V2)/dict.ltr.txt   
#    am_prefix=$(echo $am | cut -d '_' -f 1)
#    if [ $am_prefix = 'w2v2' ]; then
#        for layer in $(echo $layers | sed 's/,/ /g'); do
#            for x in train test; do
#                python scripts/extract_wav2vec_force_alignment.py \
#                   $tgt_dir/$am/feat/layer$layer $W2V2_ALIGN_DIR \
#                    --vocab-file $vocab_file \
#                    --split $x \
#                    --save-dir $tgt_dir/$am/feat/layer$layer
#            done
#        done
#    fi
fi

echo prepare_als.sh stage 5: extract phoneme-segmented speech features
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    for x in train test; do
        for layer in $(echo $layers | sed 's/,/ /g'); do
            python scripts/merge_clusters.py \
                $tgt_dir/$am/feat/layer$layer \
                --split $x \
                --save-dir $tgt_dir/$am/feat_phn_segmented/layer$layer \
                --cluster-dir $tgt_dir/$am/feat/layer$layer \
                --pooling mean
        done
    done
fi

echo prepare_als.sh stage 6: extract phoneme-segmented longitudinal speech features
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    for x in train test; do
        for layer in $(echo $layers | sed 's/,/ /g'); do
            python scripts/create_longitudinal_sequence.py \
                $tgt_dir/$am/feat_phn_segmented/layer$layer \
                --split $x \
                --save-dir $tgt_dir/$am/feat_phn_segmented_$pooling/layer$layer \
                --pooling $pooling
        done
    done
fi
