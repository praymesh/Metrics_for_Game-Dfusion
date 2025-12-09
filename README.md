# common_metrics_on_video_quality

## Introduction 

You can easily calculate the following video quality metrics for video generation and video preiction tasks:

- **FVD**: Frechét Video Distance
- **SSIM**: Structural Similarity Index Measure
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **PSNR**: Peak-Signal-to-Noise Ratio

## Installation

Version reference: numpy-1.26.4 opencv-python-4.10.0.84 scipy-1.13.1 tqdm-4.67.1 einops-0.8.0

```
conda create -n test python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install "numpy<2.0"
pip install opencv-python tqdm einops
git clone https://github.com/JunyaoHu/common_metrics_on_video_quality
```

## Evaluation Example

For example, we evaluated 8 pairs of videos, each with 30 frames, 3 channels, and a size of 64x64.

```
original  video: [8x30x3x64x64] pixel in [0,1]
generated video: [8x30x3x64x64] pixel in [0,1]
```

Run the following code `demo.py` to calculate the four metrics.

```
cd common_metrics_on_video_quality
python demo.py
```

The key content of `demo.py` is:

```
calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)
calculate_ssim(videos1, videos2, only_final=True)
calculate_psnr(videos1, videos2, only_final=True)
calculate_lpips(videos1, videos2, device, only_final=True)
```

In the example, a all-zero matrix of [8x30x3x64x64] and a all-one matrix of [8x30x3x64x64], their $\mathrm{FVD}(\mathit{frames_A}, \mathit{frames_B})$ is about 151 (Due to different pytorch versions, the number may fluctuate around 1). 

```
{
    "fvd":   {"value": [151.25648496845326]},
    "ssim":  {"value": [9.999000099990664e-05],"value_std": [0.0]},
    "psnr":  {"value": [0.0],"value_std": [0.0]},
    "lpips": {"value": [0.8140090703964233],"value_std": [0.0]}
}
```

If we set `only_final=False`,

```
calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False)
calculate_ssim(videos1, videos2, only_final=False)
calculate_psnr(videos1, videos2, only_final=False)
calculate_lpips(videos1, videos2, device, only_final=False)
```

We can calculate:

- FVD: $\mathrm{FVD}_n(\mathit{frames_A}, \mathit{frames_B}) = \mathrm{FVD}(\mathit{frames_A}[:n], \mathit{frames_B}[:n]), 10 \le n \le T $
- PSNR/SSIM/LPIPS: $\mathrm{F}_n(\mathit{frames_A}, \mathit{frames_B}) = \mathrm{avg}(\mathrm{F}(\mathit{frames_A}[i], \mathit{frames_B}[i])), \mathrm{F}=\{\mathrm{PSNR}, \mathrm{SSIM}, \mathrm{LPIPS}\}, 0 \le i \le n - 1, 1 \le n \le T $


```
{
    "fvd": {
        "value": [
            569.2296293622766,
            486.3584254441098,
            551.9610501807822,
            146.36638178542628,
            172.85453222258292,
            133.70311962583372,
            152.91750309134142,
            357.7402855012116,
            382.4668646785276,
            306.73840379649727,
            338.4151811780684,
            78.17255931098194,
            82.33446642508818,
            64.5885972265882,
            66.06281151704604,
            314.4803706985065,
            316.6870909853734,
            288.97196946254184,
            287.7805184515251,
            152.1524775765185,
            151.2564750365302
        ]
    },
    "ssim":  {"value": [9.999000099990664e-05,...,],"value_std": [0.0,...]},
    "psnr":  {"value": [0.0,...],...},
    "lpips": {"value": [0.8140090703964233,...],...}
}
```

## Comparison with orginal Tensorflow FVD metric

If you want to use the original version of FVD which comes from Tensorflow, and compare FVD result with this repo's implementation:

You should create a tensorflow-1.0 envrironment:

```
# https://github.com/universome/fvd-comparison/blob/master/requirements.txt
conda create -n tf1 python=3.7
pip install tensorflow==1.15.0 tensorflow-gan==1.0.0.dev0 tensorflow-hub==0.12.0 scipy==1.7.3 tqdm

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
calculate_fvd_tensorflow.py

# calculate_fvd(videos1, videos2, only_final=True)
# output:
# [fvd-tensorflow] [151.39244]
```




