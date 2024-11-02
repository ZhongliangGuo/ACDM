# Aesthetic Color Distance Metric

This project is for the ACDM we proposed in the paper "[Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack](https://arxiv.org/abs/2401.09673)", which is accepted by **50th European Conference on Artificial Intelligence (ECAI 2024)**.

The `eval.py` contains the settings for reproducing the result of `Table 1` in the main paper.

We use the [EFDM](https://arxiv.org/abs/2203.07740) (the pre-trained vgg and decoder can be found in the link) to generate the image pairs which have:

1. same content but different style
2. same style but different content

For datasets, we use the [COCO](https://cocodataset.org/#home) and [WikiArt (painter by numbers)](https://www.kaggle.com/c/painter-by-numbers). As some images are too huge to be processed by PIL, so I filtered some files, the files we kept for the project are listed in the `style.csv` and `content.csv`.

We used 10,000 pairs of images with `random_seed=3407`, the result is:

| IQA       | positive pair | negative pair |
| --------- | ------------- | ------------- |
| SSIMc     | 0.2871        | 0.4072        |
| LPIPS_vgg | 0.5851        | 0.5459        |
| ACDM      | 0.0464        | 0.2982        |


## Enviroment

```
lpips==0.1.4
numpy==2.0.0
Pillow==9.2.0
torch==2.0.0
torchvision==0.15.1
tqdm==4.65.0
```


## Cite

```latex
@incollection{guo2024artwork,
  title={Artwork protection against neural style transfer using locally adaptive adversarial color attack},
  author={Guo, Zhongliang and Dong, Junhao and Qian, Yifei and Wang, Kaixuan and Li, Weiye and Guo, Ziheng and Wang, Yuheng and Li, Yanli and Arandjelovi{\'c}, Ognjen and Fang, Lei},
  booktitle={ECAI 2024},
  pages={1414--1421},
  year={2024},
  publisher={IOS Press}
}
```

