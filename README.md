# hullseek

A deep learning computer vision model to detect and classify biofouling intensity on submerged structures (e.g. underwater components of ships).

## Summary and goals

Computer vision models are now excellent at many classification tasks, but underwater images remain challenging for a variety of reasons. 
Images taken underwater are generally much lower quality, and much less plentiful, so good datasets are rare. 
One key problem motivating underwater image recognition is the detection of biofouling on underwater components of ships as a part of maintenance.
Here we present a computer vision model pipeline based on YOLO (You Only Look Once) for automatic detection and estimation of the extent of biofouling present in an underwater image.
In order to utilize YOLO, we also developed a data processing pipeline involving SAM3 for adding bounding boxes and segmentation.

## Data sets

We utilized two data sets:

1. Biofouling dataset, a collection of ~10,000 underwater images labeled with simplified level of fouling (SLoF) 0, 1, or 2. 
This data set is described in [Automating the assessment of biofouling in images using expert agreement as a gold standard](https://www.nature.com/articles/s41598-021-81011-2), 
and some example images are available at [GitHub](https://github.com/emannix/automating-the-assessment-of-biofouling/tree/main). The data is heavily class imbalanced, with over 7000 entries in class 0.

2. Fouling detection dataset of ~800 images with bounding box annotations formatted for YOLO, available on [Roboflow](https://universe.roboflow.com/abdul-vdimo/fouling-detection-m3h28). 
Each bounding box is labelled with one of 7 classes such as clean, rust, starfish, barnacles, etc.

## Baseline random forest classifier

We divded all 10,000 images in data set 1 into an 80/10/10 train/test/validation, and trained two baseline models for classifying SLoF: a dummy classifier and random forest classifier. 
Because of class imbalance, the dummy classifier has relatively high accuracy (0.77) but is terrible on every other metric.

The random forest classifier is based on the original data features of vessel.id, niche.area, and paint.quality, as well as multiple engineered complexity-type features: 
Shannon entropy, gradient energy, Laplacian entropy, and variances of each of the red, green, and blue channels. 
To be explicit, the random forest classifier is not actually looking at the image file in any way, it is only based upon various numerical features computed from the image.
The random forest was an ensemble of 500 trees, with max depth of 10, and balanced classes to account for class imbalance.
As expected, this random forest classifier is very imprecise. Surprisingly, the highest feature as ranked by feature importance score was the paint.quality feature present in the original data set.

We also briefy experimented with some ResNet and EfficientNet based models, but did not obtain any significant results so we abandoned these to pursue YOLO.

## Data augmentation and cleaning, SAM3

To improve on the baseline, we turned to YOLO, a powerful pretrained computer vision model. 
However, in order to use YOLO, we needed data in a particular format, specifically data including certain kinds of bounding boxes.
Data set 2 already has these bounding boxes, but data set 1 does not. In order to take advantage of the larger size of data set 1,
we developed a pipeline for processing images in data set 1 to add bounding boxes, utilizing SAM3 (Segment Anything Model). 

First, we sorted images in data set 1 into SLoF level 0 (around 7000 images) and SLoF levels 1 and 2 (around 3000 images).
For level 0 (little or no biofouling), we simply set a single bounding box for the entire image.
For levels 1 and 2 (medium or  high biofouling), we first manually placed bounding boxes on around 50 images, then
with a combination of mimicry and text prompts instructed SAM3 to place bounding boxes in the remaining images.
Due to various technical limitations, this only worked for about 1400 images, so we end up with around 1400 images 
of SLoF level 2 or 3 as a sub-data set.

TO ADD: more explanation of how SAM3 does this

## YOLO

First we fine-tuned a YOLO model on data set 2 to label each bounding box as one of 7 classes (clean, starfish, barnacle, etc.). Then we merged the 6 non-clean classes as "clean."

We then pursued two directions:

A. Apply that YOLO model to the annotated data set 1 as a validation set.
B. Retrain YOLO model on full annotated data set 1, with balanced representation.

Given the reduced image set of ~1400 SLoF 1/2 images with annotated bounding boxes from biofouling (data set 1), we fine-tuned a YOLO binary classifier model on this data.

