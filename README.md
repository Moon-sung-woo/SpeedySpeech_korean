# SpeedySpeech [[Paper link]](https://arxiv.org/pdf/2008.03802.pdf)

## 수정중입니다.
- 전체 프로젝트는 docker를 사용
- pycharm에서 바로 실행이 되게 경로를 수정했습니다.
    - 터미널을 켜서 실행을 하실분들은 경로 수정해 주셔야 합니다.

## Installation instructions
The code was tested with `python 3.6.9`, `cuda 10.0.130` and `GNU bash 5.0.3` on Ubuntu 18.04.

```
git clone https://github.com/Moon-sung-woo/SpeedySpeech_korean.git
cd speedyspeech

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training
To train speedyspeech, durations of phonemes are needed.

**1. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/)** and unzip into `datasets/data/LJSpeech-1.1`

**1-1. If you want to train Korean, download the [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset/data)
한국어를 사용하실 분은 kss데이터를 다운받아 사용하세요!
한국어 데이터를 다운받기 위해서는 로그인이 필요하기 때문에 로그인 하셔서 데이터를 다운받으시고
아래와 같은경로에 압축을 풀어주시면 됩니다.
```
wget -O code/datasets/data/LJSpeech-1.1.tar.bz2 \
    https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf code/datasets/data/LJSpeech-1.1.tar.bz2 -C code/datasets/data/
```
**2. Train the duration extraction model**
```
터미널에서 실행하는 경로와 pycharm에서 바로 실행하는 경로가 다를 수 있으니
경로 확인하시고 실행해주세요
저는 pycharm에서 바로 실행되게 경로를 수정했습니다.
python code/duration_extractor.py -h  # display options
python code/duration_extractor.py \
    --some_option value
tensorboard --logdir=logs
```
**3. Extract durations from the trained model** - creates alignments.txt file in the LJSpeech-1.1 folder
```
python code/extract_durations.py logs/your_checkpoint code/datasets/data/LJSpeech-1.1 \
    --durations_filename my_durations.txt
```
**4. Train SpeedySpeech**
```
python code/speedyspeech.py -h
python code/speedyspeech.py \
    --durations_filename my_durations.txt
tensorboard --logdir=logs2
```

## Inference
**1. Download pretrained MelGAN** checkpoint
```
wget -O checkpoints/melgan.pth \
    https://github.com/seungwonpark/melgan/releases/download/v0.1-alpha/nvidia_tacotron2_LJ11_epoch3200.pt 
```

**2. Download pretrained SpeedySpeech** checkpoint from the latest release.
```
wget -O checkpoints/speedyspeech.pth \
    https://github.com/janvainer/speedyspeech/releases/download/v0.2/speedyspeech.pth 
```

**3. Run inference**
```
mkdir synthesized_audio
printf "One sentence. \nAnother sentence.\n" | python code/inference.py --audio_folder synthesized_audio
```
The model treats each line of input as an item in a batch.
To specify different checkpoints, what device to run on etc. use the following:
```
printf "One sentence. \nAnother sentence.\n" | python code/inference.py \
    --speedyspeech_checkpoint <speedyspeech_checkpoint> \
    --melgan_checkpoint <melgan_checkpoint> \
    --audio_folder synthesized_audio \
    --device cuda
```

Files wil be added to the audio folder. The model does not handle numbers. please write everything in words.
The list of allowed symbols is specified in ```code/hparam.py```. 
## License
This code is published under the BSD 3-Clause License.
1. `code/melgan` - [MelGAN](https://github.com/seungwonpark/melgan) by Seungwon Park (BSD 3-Clause License)
2. `code/utils/stft.py` - [torch-stft](https://github.com/pseeth/torch-stft) by Prem Seetharaman (BSD 3-Clause License)
3. `code/pytorch_ssim` - [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) by Po-Hsun-Su (MIT)
