# hullseek

A deep learning computer vision model to detect and classify biofouling intensity on submerged structures (e.g. underwater components of ships).

ADD HERE: More overall summary, table of contents for the readme

## Data sets

We utilized two data sets:

1. Biofouling dataset, a collection of about 10,000 underwater images described in [Automating the assessment of biofouling in images using expert agreement as a gold standard](https://www.nature.com/articles/s41598-021-81011-2), 
example images available on [GitHub](https://github.com/emannix/automating-the-assessment-of-biofouling/tree/main).

2. Fouling detection dataset, available on [Roboflow](https://universe.roboflow.com/abdul-vdimo/fouling-detection-m3h28).

## Data augmentation and cleaning

The entirey of data set 1 (biofouling) was used for training the baseline classifiers (dummy and random forest). 
In order to train/fine-tune a YOLO model, we needed very specific data structure, including bounding boxes. 
Data set 2 is already fully annotated in this way. We explored using SAM3 to automate annotation of data set 1 with bounding boxes,
but this was not possible for all images. After restricting to images where automated annotation with bounding boxes was possible,
about 6,000 images remained in data set 1.

## Models

In addition to a dummy classifier baseline, we developed the following models.

### Random forest classifier

For the biofouling data set, we did some feature engineering to explore complexity-related features for images, including Shannon entropy, gradient energy, and variance in each of the red, green, and blue spectrums. 
Based on these features, we trained a random forest classifier, which unsurprisingly performs quite poorly. The random forest was an ensemble of 500 trees, with max depth of 10, and balanced classes to account for class imbalance.

### YOLO

### SAM3

