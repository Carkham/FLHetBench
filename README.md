# FLHetBench

![framework_v8](assets/framework.png)

## Overview

- `metric.py` contains the methods for calculating **DevMC-R/T(device heterogeneity), StatMC-R/T(state heterogeneity) and InterMC-R/T(their interplay)**.
- `sampling.py` contains **DPGMM** and **DPCSM**.
- `data/` folder contains all the databases used and the sampled heterogeneous datasets.
- `bench/` folder contains benchmark framework.

## Usage

### 1. Heterogeneity simulation

#### 1.1 Prepare Heterogeneity Dataset

We use different datasets for device heterogeneity and state heterogeneity.

- dataset for device heterogeneity.
  - `data/mobiperf_tcp_down_2018.json`
  - `data/mobiperf_tcp_down_2019.json`
- dataset for state heterogeneity.
  - `data/cached_timers.json`
- Our proposed training latency data. **Our data will be dynamically updated, and we sincerely invite more people to participate. If you're interested, click on the [link](https://docs.google.com/document/d/1KwNdgW57gNs8VskZwdUGhLg6b_XaplPWdaTRcmbQeWk/edit?usp=sharing) to learn more**
  - `data/device_latency.json`

#### 1.2 Sampling

**For simplicity, you can use the device and state heterogeneity data provided in `bench/cached_sample_data` directly or build your own heterogeneous environment with DPGMM and DPCSM**

##### DPGMM (Device heterogeneity)

```python
# network speed infomations refer to 'data/device/mobiperf_tcp_down_2018.json'
speed_info = [
    {
        "tcp_speed_results": [4121, 4753.5, ...], # network speed list from Mobiperf
        ...
    },
    ...
]
n = 2466 # number of clients
mu = 6000 # expected average speeed
K = 50 # number of clusters
simga = 0. # control of divergence
random_seed = 42

_, sampled_speed_mean, sampled_speed_std, samples = DPGMM_sampling(speed_info, mu0=mu, K=k, sigma=sigma, n=2466, seed=random_seed)
```

##### DPCSM (State heterogeneity)

```python
# state score dict used for sampling by DPCSM
score_dict = {
    '681': 0.1,
    '573': 0.2,
    ...
}
n = 2466 # number of clients
alpha = 100 # control of divergence
start_rank = 0 # control of start rank, the same as $StartRank$ in the paper

# return a list of length n=2466 with elements that are keys in score_dict
samples = DPCSM_sampling(score_dict, n=2466, alpha=alpha, start_rank=start_rank)
```

#### 1.3 Metric

We use **DevMC-R/T** for assessing device heterogeneity. We use **StatMC-R/T** to assess state heterogeneity and **InterMC-R/T** for their interplays.

Please refer to [metric_example.ipynb](metric_example.ipynb) for snippets.

### 2. Prepare Image Dataset

- OpenImage
  - Download the data partition from [OpenImage](https://drive.google.com/file/d/1ZkNndV-MsMl5fsF3scIy7PNTvotjMtf2/view?usp=sharing)
  - Put the downloaded openImg.npy at sub-folder `bench/data`
- COVID-FL dataset
  - Download the data and partitions file from [COVID-FL](https://drive.google.com/file/d/1BiG30JJ7U2BT0x92DjwfPeLb-uwTHdUV/view?usp=sharing)

### 3. Download the Pre-trained Models

- Vision Transformer: download the file from [ViT-B_16](https://drive.google.com/file/d/13DD-dYP3RjMwkYeLYeqdGCXfUBZKEZcQ/view?usp=drive_link) and put it under the sub-folder `bench/checkpoint`

### 4. Run

```bash
# set the gpu No. to 0
bash shells/biasprompt.sh 0
```

## Additional Notes

Some important tags for training settings:

- `--dataset_name`: we provide implement of OpenImage and COVID-FL in main.py.
- `--aggregation_strategy`: type of sever aggregation strategy, supports ["deadline", "readiness"].
- `--deadline`: round deadline for deadline-based strategy.
- `--num_rounds`: total communication rounds for deadline-based strategy.
- `--target_acc`: target performance for readiness-based strategy.
- `--device_path`: the sampled heterogeneous device data file path.
- `--state_path`: the sampled heterogeneous state data file path.

## BiasPrompt+

![biasprompt+_v5](assets/biasprompt+.png)

BiasPrompt+ comprises two modules: a gradient surgery-based staleness-aware aggregation strategy (`bench/helpers/gradient_surgery_helper.py`) and a communication-efficient module BiasPrompt (`bench/models/BiasPrompt.py`) based on fast weights.

## Acknowledgments

- Our code is based on [PKU-Chengxu/FLASH (github.com)](https://github.com/PKU-Chengxu/FLASH)

- ResNet50 and ViT implementations are based on https://github.com/rwightman/pytorch-image-models and [vpt](https://github.com/KMnP/vpt)

