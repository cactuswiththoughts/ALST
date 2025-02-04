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
#export ALIGN_DIR=/data/sls/scratch/nauman/shared/LimingsWork/ALS_EXPS/ALS_model_adapt_dumps__Exp1_modelDownloaded_dataALS__noOOV/out_TextGrids
export ALIGN_DIR=/data/sls/scratch/nauman/shared/LimingsWork/ALS_EXPS/ALS_model_adapt_dumps__Exp2_modelFHS83Finetuned_dataALS/out_TextGrids

am_name=whisper_medium 
#am_name=whisper_base 
#am_name=whisper_large-v2
start_layer=0
end_layer=23
#end_layer=31
layers=
for ((i=start_layer; i<=end_layer; i++)); do
    layers=$layers,$i
done
layers=${layers#,}
echo $layers

pooling=mean+concat
label=vieira
model=alst

if [ $model = alst_encdec ]; then
    depth=2
    lr=1e-4
else
    depth=2
    lr=1e-4
fi
batch_size=25
embed_dim=512

tgt_dir=$PWD/../data/als
if [ $label = vieira ]; then
    setup=vieira
else
    setup=with_$label
fi

stage=0
stop_stage=1
echo stage 0: Speech feature extraction
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    bash scripts/prepare_als.sh $tgt_dir $layers $am_name $pooling $label
fi

echo stage 1: Train and test ALS predictor
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    am=$(echo $am_name | cut -d'_' -f 2)
    exp_dir=$PWD/../exp/alst_whisper-${am}_seed42

    for x in train test; do
        python traintest.py --mode eval --data-dir $tgt_dir/$setup/${am_name}/feat_${pooling} --test-split $x \
  	        --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
	        --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name

        python scripts/extract_scores_vs_days.py \
            --in-path $tgt_dir/$setup/${am_name}/feat_${pooling}/layer0/$x.ids \
            --score-path $exp_dir/preds/${x}_pred_als_scores.npy \
            --out-path $exp_dir/${x}_${model}_${am_name}_score_vs_days.csv

        python scripts/extract_scores_vs_days.py \
            --in-path $tgt_dir/$setup/${am_name}/feat_${pooling}/layer0/$x.ids \
            --score-path $exp_dir/preds/${x}_gold_als_label.npy \
            --out-path $exp_dir/${x}_score_vs_days.csv
    done
fi

echo stage 2: Train and test ALS predictor using different regularizer weights
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    mse_weight=1.0
    for ce_weight in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
     	 exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-${setup}-with_mask-ce_weight${ce_weight}-mse_weight${mse_weight}
	    python traintest.py --mode eval --data-dir $tgt_dir/$setup/${am_name}/feat_${pooling} \
	    --layers $layers --lr $lr --exp-dir $exp_dir --depth $depth \
	    --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name \
	    --ce_weight $ce_weight --mse_weight $mse_weight
    done

    # Extract results with different regularizer weights
    cp -r ../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}-${pooling}-${setup}-with_mask-ce_weight1.0-mse_weight${mse_weight} \
        ../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-${setup}-with_mask-ce_weight1.0-mse_weight${mse_weight}

    python scripts/extract_ablation_results.py \
        --exp-name ../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}_${end_layer}-${pooling}-${setup}-with_mask-ce_weight{}-mse_weight${mse_weight} \
        --hp-name "CE weight" --hps 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
        --out_file ${model}_${am_name}_${setup}_ce-weights_mse-weight${mse_weight}_results.csv
fi

echo stage 3: Train and test ALS predictor using different AM layers
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    ce_weight=1.0
    mse_weight=1.0
    # XXX
    for layer in $(seq $start_layer $end_layer); do
        exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${layer}-${pooling}-${setup}-with_mask-ce_weight${ce_weight}-mse_weight${mse_weight}
        python traintest.py --mode eval --data-dir $tgt_dir/$setup/$am_name/feat_${pooling} \
            --layers $layer --lr $lr --exp-dir $exp_dir --depth $depth \
            --batch_size $batch_size --embed_dim $embed_dim --model $model --am $am_name \
	        --ce_weight $ce_weight --mse_weight $mse_weight
    done

#    python scripts/extract_layerwise_results.py \
#        --exp-name ../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer{}-${pooling}-${setup}-with_mask-ce_weight${ce_weight}-mse_weight${mse_weight} \
#        --layers $layers \
#        --out_file ${model}_${am_name}_${setup}_ce-weight${ce_weight}_mse-weight${mse_weight}_layerwise_results.csv
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    ce_weight=1.0
    mse_weight=1.0
    exp_dir=../exp/${model}-${lr}-${depth}-${batch_size}-${embed_dim}-${am_name}-layer${start_layer}-${pooling}-${setup}-with_mask-ce_weight${ce_weight}-mse_weight${mse_weight}/preds
    echo $exp_dir
    python scripts/extract_scores_vs_days.py \
        --in-path $tgt_dir/$setup/${am_name}/feat_${pooling}/layer0/test.ids \
        --score-path $exp_dir/pred_als_scores.npy \
        --out-path test_${model}_${am_name}_score_vs_days.csv

    python scripts/extract_scores_vs_days.py \
        --in-path $tgt_dir/$setup/${am_name}/feat_${pooling}/layer0/test.ids \
        --score-path $exp_dir/gold_als_label.npy \
        --out-path test_score_vs_days.csv
fi
