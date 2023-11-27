#!/bin/bash
#SBATCH -J als_whisper
#SBATCH -o logs/%j_als_whisper.out
#SBATCH -e logs/%j_als_whisper.err
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --qos=regular

source /data/sls/scratch/limingw/miniconda3/etc/profile.d/conda.sh
PYTHON_VIRTUAL_ENVIRONMENT=/data/sls/scratch/limingw/miniconda3/envs/whisper
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
export FAIRSEQ_ROOT=/data/sls/scratch/limingw/fairseq-0.12.1
export ALS_WAV_DIR=/data/sls/scratch/yuangong/dataset/ALS/Voice_Recordings_16k_reorder
export TRITON_CACHE_DIR=$PYTHON_VIRTUAL_ENVIRONMENT/.triton_cache

am_name=whisper_medium 
#am_name=whisper_base 
#am_name=whisper_large-v2
#start_layer=0
#start_layer=6
start_layer=23
#start_layer=31
#end_layer=6
end_layer=23
#end_layer=31
layers=
for ((i=start_layer; i<=end_layer; i++)); do
    layers=$layers,$i
done
layers=${layers#,}
echo $layers

pooling=mean+concat
#label=text
label=score
model=alst
#model=alst_encdec
if [ $model = alst_encdec ]; then
    depth=2
    lr=1e-4
else
    depth=2
    lr=1e-3
fi
batch_size=25
embed_dim=512
# embed_dim=24

tgt_dir=$(pwd)/../data/als

stage=3
stop_stage=3
echo stage 0: Speech feature extraction
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    bash scripts/prepare_als.sh $tgt_dir $layers $am_name $pooling $label
fi

echo stage 1: Train and test ALS predictor
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-with_$label-with_mask
    python traintest.py --data-dir $tgt_dir/with_$label/${am_name}/feat_${pooling} \
    --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
    --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name 

 #   exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-marco-with_mask
 #   python traintest.py --data-dir $tgt_dir/marco/${am_name}/feat_${pooling} \
 #   --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
 #   --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name 
fi

echo stage 2: Train and test ALS predictor using different AM layers
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for layer in $(seq $start_layer $end_layer); do
        exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${layer}-${pooling}-with_$label-with_mask
#        python traintest.py --data-dir $tgt_dir/with_$label/${am_name}/feat_${pooling} \
#        --layers $layer --lr $lr --exp-dir $exp_dir --depth $depth \
#        --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name
    done

    python scripts/extract_layerwise_results.py \
        --exp-name ../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer{}-${pooling}-with_$label-with_mask \
        --layers $layers \
        --out_file ${model}_${am_name}_with_${label}_layerwise_results.csv
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-with_$label-with_mask/preds/pred_als_label.npy
    echo $exp_dir
    python scripts/extract_scores_vs_days.py \
        --in-path $tgt_dir/with_$label/${am_name}/feat_${pooling}/layer0/test.ids \
        --score-path $exp_dir \
        --out-path test_${model}_${am_name}_score_vs_days.csv
fi
