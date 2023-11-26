#!/bin/bash

tgt_dir=$1
layers=$2
am=$3
pooling=$4
label=$5
# XXX tgt_dir=$tgt_dir/with_$label
tgt_dir=$tgt_dir/marco

stage=1
stop_stage=2

if [ ! -d $tgt_dir ]; then 
    mkdir -p $tgt_dir 
fi

echo prepare_als.sh stage 0: extract metadata for ALS
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    python scripts/prepare_marco_data.py \
        --in-dir $tgt_dir --out-dir $tgt_dir \
        --wav-dir $ALS_WAV_DIR

#    label_dir=$ALS_WAV_DIR/../Metadata
#    python scripts/prepare_data.py \
#        $ALS_WAV_DIR $label_dir $tgt_dir \
#        --with-labels $label
#    python scripts/convert_kaldi_to_fairseq.py \
#        --in-dir $tgt_dir \
#        --out-dir $tgt_dir --test-speaker-ratio 0.2

#    KALDI_ROOT=/data/sls/scratch/share-202102/kaldi
#        --in-dir $KALDI_ROOT/egs/gop_als/s5/data/all \
fi

echo prepare_als.sh stage 1: extract speech features
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for x in train test; do 
        am_prefix=$(echo $am | cut -d '_' -f 1)
        am_suffix=$(echo $am | cut -d '_' -f 2)
        echo $layers
        if [ $am_prefix = 'w2v2' ]; then
            python scripts/wav2vec_extract_features.py \
                $tgt_dir --split $x --save-dir $tgt_dir/${am}/feat --checkpoint $W2V2 --layers $layers
        elif [ $am_prefix = 'whisper' ]; then
            python scripts/extract_whisper_feats.py \
                $tgt_dir --split $x --save-dir $tgt_dir/${am}/feat --model-name $am_suffix --layers $layers
        fi
    done
fi

echo prepare_als.sh stage 2: extract longitudinal speech features
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for x in train test; do
        for layer in $(echo $layers | sed 's/,/ /g'); do
            python scripts/create_longitudinal_sequence.py \
                $tgt_dir/${am}/feat/layer${layer} \
                --split $x \
                --save-dir $tgt_dir/${am}/feat_${pooling}/layer${layer} \
                --pooling $pooling
        done
    done
fi

echo prepare_als.sh stage 3: extract GOP speech features
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    vocab_file=$(dirname $W2V2)/dict.ltr.txt
    am_prefix=$(echo $am | cut -d '_' -f 1)
    if [ $am_prefix = 'w2v2' ]; then
        for x in train test; do
            python scripts/extract_wav2vec_gop.py $tgt_dir/${am}_layer-1 \
                $W2V2_ALIGN_DIR \
                --vocab-file $vocab_file \
                --split $x \
                --save-dir $tgt_dir/../with_${label}_align/${am}_layer-1_gop
        done
    fi
fi

echo prepare_als.sh stage 4: extract longitudinal GOP speech features
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    for x in train test; do
        python scripts/create_longitudinal_sequence.py \
            $tgt_dir/../with_${label}/${am}_layer-1_gop \
            --split $x \
            --save-dir $tgt_dir/../with_${label}/${am}_layer-1_gop_${pooling} \
            --pooling $pooling
    done
fi

echo prepare_als.sh stage 5: extract phoneme-segmented speech features
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    for x in train test; do
        python scripts/merge_clusters.py \
            $tgt_dir/../with_${label}/${am}_layer${layer} \
            --split $x \
            --save-dir $tgt_dir/../with_${label}/${am}_layer${layer}_phn_segmented \
            --cluster-dir $tgt_dir/../with_${label}/${am}_layer-1_gop \
            --pooling mean
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    for x in train test; do
        python scripts/create_longitudinal_sequence.py \
            $tgt_dir/../with_${label}/${am}_layer${layer}_phn_segmented \
            --split $x \
            --save-dir $tgt_dir/../with_${label}/${am}_layer${layer}_phn_segmented_${pooling} \
            --pooling $pooling
    done
fi


