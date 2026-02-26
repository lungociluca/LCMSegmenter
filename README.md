# Leveraging Latent Consistency Models for training-free, open-vocabulary semantic segmentation

This work is based on the repository [here](https://github.com/VCG-team/DiffSegmenter?tab=readme-ov-file), the official implementation of the paper [Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter](https://arxiv.org/abs/2309.02773).

# My contribution

Extending the work of DiffSegmenter to Latent Consistency Models pipelines, for improved robustness across different choices for the Diffusion model family parameter: denoising timestamp. Updates to the original repository also allows: testing multiple configurations of denoising timestamp with a single run, run the code on batches instead of one image-class pair at a time, running the prompts and masks computing steps separately and performing validation and testing on disjoint data sets.

### Requirements

* Linux, CUDA>=11.7, GCC>=9.4
  
* Python>=3.8

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n ldm python=3.8
    ```
    Then, activate the environment:
    ```bash
    conda activate ldm
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Configurations setup

Copy the file local_config_template.py into local_config.py, then fill in paths and configurations.

### Dataset preparation

Please download datasets and organize them as following:

```

├── COCO2014
│   ├── annotations
│   ├── coco_seg_anno
│   ├── images
│   │   ├── test2014
│   │   ├── train2014
│   │   └── val2014
│   └── mask
│       ├── train2014
│       └── val2014


└── VOCdevkit
    ├── VOC2010
    │   ├── Annotations
    │   ├── ImageSets
    │   │   ├── Action
    │   │   ├── Layout
    │   │   ├── Main
    │   │   ├── Segmentation
    │   │   └── SegmentationContext
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   ├── SegmentationContext
    │   └── SegmentationObject
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation
        ├── JPEGImages
        ├── SegmentationClass
        ├── SegmentationClass
        └── SegmentationObject
```

### Open Vocabulary Semantic Segmentation

#### Evaluation

For the setting of Open Vocabulary Semantic Segmentation， our model does not require training; it directly produces segmentation results.


The ‘open_vocabulary’ folder contains code for open vocabulary semantic segmentation. It includes scripts for the voc, coco, and Pascal context datasets.

Taking the voc10 dataset as an example:

Step 1: Modify your dataset path in the Python file.

Step 2: Run ptp_stable_voc10.py to generate segmentation results.

```
python ptp_stable_voc10.py
```

Step 3: Run the evaluation script, remember to update the file path. MIoU will be recorded in eval.txt

```
python evaluation_voc10.py
```
