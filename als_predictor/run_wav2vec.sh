#!/bin/bash
#SBATCH -J als_wav2vec2
#SBATCH -o logs/%j_als_wav2vec2.out
#SBATCH -e logs/%j_als_wav2vec2.err
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

source /data/sls/scratch/limingw/miniconda3/etc/profile.d/conda.sh
PYTHON_VIRTUAL_ENVIRONMENT=/data/sls/scratch/limingw/miniconda3/envs/fairseq
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
export FAIRSEQ_ROOT=/data/sls/scratch/limingw/fairseq-0.12.1
export ALS_WAV_DIR=/data/sls/scratch/yuangong/dataset/ALS/Voice_Recordings_16k_reorder
export W2V2=/data/sls/scratch/limingw/models/finetuned/wav2vec_big_960h.pt
export W2V2_ALIGN_DIR=/data/sls/scratch/yuangong/als/w2v_align

am_name=w2v2_big_960h
#start_layer=0
start_layer=14
end_layer=14
#end_layer=23
layers=
for ((i=start_layer; i<=end_layer; i++)); do
    layers=$layers,$i
done
layers=${layers#,}
echo $layers

label=text_align 
#label=score
#label=text
#pooling=mean+concat
#pooling=concat
pooling=concat

model=alst
#model=alst_encdec
#model=linear
if [ $model = alst_encdec ]; then
    depth=2
    lr=1e-3
else
    depth=2
    lr=1e-3
fi
batch_size=12
embed_dim=512
# embed_dim=24

tgt_dir=$(pwd)/../data/als
if [ $label = vieira ]; then
    setup=vieira
else
    setup=with_$label
fi

stage=1
stop_stage=1
echo stage 0: Speech feature extraction
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    bash scripts/prepare_als.sh $tgt_dir $layers $am_name $pooling $label
fi

echo stage 1: Train and test ALS predictor
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    pooling=phn_segmented_$pooling
    exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-${pooling}-layers${start_layer}_${end_layer}-$setup
    python traintest.py --data-dir $tgt_dir/$setup/$am_name/feat_$pooling \
    --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
    --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name

#    exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-${pooling}-layers${start_layer}_${end_layer}-$setup-use_phn_label
#    python traintest.py --data-dir $tgt_dir/$setup/$am_name/feat_$pooling \
#    --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
#    --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name --use-phn-labels

#   python traintest.py --data-dir $tgt_dir/$setup/${am_name}_layer${layer}_${pooling} \
#   --lr $lr --exp-dir ${exp_dir}-no_longitudinal --depth $depth \
#   --batch_size $batch_size --embed_dim $embed_dim --am $am_name --no-longitudinal

#    exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-marco-with_mask
#    python traintest.py --data-dir $tgt_dir/marco/${am_name}/feat_${pooling} \
#    --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
#    --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name 
fi

echo stage 2: Train and test ALS predictor using different AM layers 
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for layer in $(seq $start_layer $end_layer); do
        exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${layer}-${pooling}-$setup-with_mask
        python traintest.py --data-dir $tgt_dir/$setup/${am_name}/feat_${pooling} \
        --layers $layer --lr $lr --exp-dir $exp_dir --depth $depth \
        --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name
    done

    python scripts/extract_layerwise_results.py \
        --exp-name ../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer{}-${pooling}-$setup-with_mask \
        --layers $layers \
        --out_file ${model}_${am_name}_$setup_layerwise_results.csv
fi
