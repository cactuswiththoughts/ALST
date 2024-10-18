# Automatic Prediction of Amyotrophic Lateral Sclerosis Progression using Longitudinal Speech Transformer

*Amyotrophic Lateral Sclerosis Transformer* (ALST) is a neural network-based predictor of amyotrophic lateral sclerosis (ALS) progression leveraging both patient voice and longitudinal information.
If you find this project useful, please consider citing our paper:
~~~~
@inproceedings{wang2024alst,
  author    = {Liming Wang and Yuan Gong and Nauman Dawalatabad and Marco Vilela and Katerina Placek and Brian Tracey and Yishu Gong and Fernando Vieira and James Glass},
  title     = {Automatic Prediction of Amyotrophic Lateral Sclerosis Progression using Longitudinal Speech Transformer},
  booktitle = {Interspeech 2024},
  year      = {2024}
}
~~~~

## How to run it
### Key dependencies
torch >= 1.13.1
[fairseq](https://github.com/pytorch/fairseq) >= 1.0.0 (for wav2vec 2.0 based models only)
[openai-whisper](https://github.com/openai/whisper) >= 20231117

### Data preparation
To obtain the ALS data used for training, please contact [ALS-TDI](https://www.als.net/) and use the split provided in data/. For inference using your own data, create a directory $tgt_dir containing the following files:
- {train, test}.tsv (please don't forget "./" in the first line):
  ~~~
  ./
  /path/to/your/second/wav/dir/for/patient/1/filename1.wav
  /path/to/your/second/wav/dir/for/patient/1/filename2.wav
  ...
  /path/to/your/second/wav/dir/for/patient/2/filename1.wav
  ...
  ~~~
- {train, test}.score: each line contains the label for each wav file in the same order as in .tsv
- {train, test}.days (optional): each line contains the number of days of each recording since ALS diagnosis of the patient

Then, run the data preparation code:
~~~
  bash scripts/prepare_als.sh $tgt_dir $layers whisper-large mean+concat score
~~~
where $layers is a comma-separated list of layer numbers of the backbone, e.g. Whisper.

### Training
~~~~
  cd als_predictor

  # ALST with wav2vec 2.0 backbone
  bash run_wav2vec.sh

  # ALST with Whisper backbone
  bash run_whisper.sh

  # SVM baselines
  bash run_{wav2vec,whisper}_svm.sh
~~~~
### Inference
~~~~
cd als_predictor
$tgt_dir/$setup/$am_name/feat_${pooling}
python traintest.py --mode eval --data-dir $feat_dir/feat_mean+concat \
    --exp-dir ./exp/alst_whisper-large-v2_seed32 --batch_size 16 --model alst --am whisper-large \
~~~~
