<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (ICLR'24) TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting </b></h2>
</div>


<div align="center">

![](https://img.shields.io/github/last-commit/KimMeen/Time-LLM?color=green)
![](https://img.shields.io/github/stars/kwuking/TimeMixer?color=yellow)
![](https://img.shields.io/github/forks/kwuking/TimeMixer?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>


---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@inproceedings{wang2023timemixer,
  title={TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author={Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y and ZHOU, JUN},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Introduction
🏆 **TimeMixer**, as a fully MLP-based architecture, taking full advantage of disentangled multiscale time series, is proposed to **achieve consistent SOTA performances in both long and short-term forecasting tasks with favorable run-time efficiency**.

🌟**Observation 1: History Extraction** 

Given that seasonal and trend components exhibit significantly different characteristics in time series, and different scales of the time series reflect different properties, with seasonal characteristics being more pronounced at a fine-grained micro scale and trend characteristics being more pronounced at a coarse macro scale, it is therefore necessary to decouple seasonal and trend components at different scales.

<p align="center">
<img src="./figures/motivation1.png"  alt="" align=center />
</p>

🌟**Observation 2: Future Prediction** 

Integrating forecasts from different scales to obtain the final prediction results, different scales exhibit complementary predictive capabilities.

<p align="center">
<img src="./figures/motivation2.png"  alt="" align=center />
</p>

## Overall Architecture
TimeMixer as a fully MLP-based architecture with **Past-Decomposable-Mixing (PDM)** and **Future-Multipredictor-Mixing (FMM)** blocks to take full advantage of disentangled multiscale series in both past extraction and future prediction phases. 

<p align="center">
<img src="./figures/overall.png"  alt="" align=center />
</p>

### Past Decomposable Mixing 
we propose the **Past-Decomposable-Mixing (PDM)** block to mix the decomposed seasonal and trend components in multiple scales separately. 

<p align="center">
<img src="./figures/past_mixing1.png"  alt="" align=center />
</p>

Empowered by seasonal and trend mixing, PDM progressively aggregates the detailed seasonal information from fine to coarse and dive into the macroscopic trend information with prior knowledge from coarser scales, eventually achieving the multiscale mixing in past information extraction.

<p align="center">
<img src="./figures/past_mixing2.png"  alt="" align=center />
</p>

### Future Multipredictor Mixing 
Note that **Future Multipredictor Mixing (FMM)** is an ensemble of multiple predictors, where different predictors are based on past information from different scales, enabling FMM to integrate complementary forecasting capabilities of mixed multiscale series.

<p align="center">
<img src="./figures/future_mixing.png"  alt="" align=center />
</p>



## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
bash ./scripts/long_term_forecast/ECL_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Traffic_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Solar_script/TimeMixer.sh
bash ./scripts/long_term_forecast/Weather_script/TimeMixer.sh
bash ./scripts/short_term_forecast/M4/TimeMixer.sh
bash ./scripts/short_term_forecast/PEMS/TimeMixer.sh
```

## Main Results
We conduct extensive experiments to evaluate the performance and efficiency of TimeMixer, covering long-term and short-term forecasting, including 18 real-world benchmarks and 15 baselines.

### Long-term Forecasting

<p align="center">
<img src="./figures/long_results.png"  alt="" align=center />
</p>

### Short-term Forecasting: Multivariate data

<p align="center">
<img src="./figures/pems_results.png"  alt="" align=center />
</p>

###  Short-term Forecasting: Univariate data

<p align="center">
<img src="./figures/m4_results.png"  alt="" align=center />
</p>


## Model Abalations

## Model Efficiency
TimeMixer achieves favorable efficiency in comparing with Transformer-based models.

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

## Contact

If you have any questions or want to use the code, feel free to contact:
* Shiyu Wang (kwuking@163.com or weiming.wsy@antgroup.com)
* Haixu Wu (wuhx23@mails.tsinghua.edu.cn)