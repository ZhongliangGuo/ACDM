# Aesthetic Color Distance Metric

This project is for the ACDM we proposed in our [paper](https://arxiv.org/abs/2401.09673).

The `eval.py` contains the settings for reproducing the result of `Table 1` in the main paper.

We use the [EFDM](https://arxiv.org/abs/2203.07740) (the pre-trained vgg and decoder can be found in the link) to generate the image pairs which have:

1. same content but different style
2. same style but different content

For datasets, we use the [COCO](https://cocodataset.org/#home) and [WikiArt (painter by numbers)](https://www.kaggle.com/c/painter-by-numbers). As some images are too huge to be processed by PIL, so I filtered some files, the files we kept for the project are listed in the `style.csv` and `content.csv`.