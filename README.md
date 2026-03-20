# hullseek

A deep learning computer vision model to detect and classify biofouling intensity on submerged structures (e.g. underwater components of ships).

ADD HERE: More overall summary, table of contents for the readme

## Data sets

We utilized two data sets:

1. Biofouling dataset, a collection of about 10,000 underwater images labelled with simplified level of fouling (SLoF) described in [Automating the assessment of biofouling in images using expert agreement as a gold standard](https://www.nature.com/articles/s41598-021-81011-2), 
example images available on [GitHub](https://github.com/emannix/automating-the-assessment-of-biofouling/tree/main). After processing these and adding bounding boxes to utilize YOLO, we obtained about 1,400 images.

2. Fouling detection dataset of ~800 images with bounding box annotations formatted for YOLO, available on [Roboflow](https://universe.roboflow.com/abdul-vdimo/fouling-detection-m3h28). Bounding boxes are labelled with 7 classes such as clean surface, starfish, barnacles, etc.

## Data augmentation and cleaning, SAM3

The entirey of data set 1 (biofouling) was used for training the baseline classifiers (dummy and random forest). 

In order to fine-tune a YOLO model, we needed specifically formatted data including bounding boxes with labelling. 
We used SAM3 text prompts to automate the process of adding bounding boxes to images from data set 1. 
We did this for all SLoF level 1 and 2 images in biofouling, which was successful for around 60% of the SLoF 1/2 images in biofouling, 
resulting in around 1,400 bounding-box-annotated images as a subset of the original biofouling dataset 1.

Data set 2 is already fully annotated in accordance with YOLO requirements. We explored using SAM3 to automate annotation of data set 1 with bounding boxes,
but this was not possible for all images. 

TO DO: Talk about SAM3

## Models

In addition to a dummy classifier baseline, we developed the following models.

### Random forest classifier

For the biofouling data set, we did some feature engineering to explore complexity-related features for images, including Shannon entropy, gradient energy, and variance in each of the red, green, and blue spectrums. 
Based on these features, we trained a random forest classifier, which unsurprisingly performs quite poorly. The random forest was an ensemble of 500 trees, with max depth of 10, and balanced classes to account for class imbalance.

### ResNet and EfficientNet

TO DO: Explain a bit about these

### YOLO

First we fine-tuned a YOLO model on data set 2 to label each bounding box as one of 7 classes (clean, starfish, barnacle, etc.). Then we merged the 6 non-clean classes as "clean."

We then pursued two directions:

A. Apply that YOLO model to the annotated data set 1 as a validation set.
B. Retrain YOLO model on full annotated data set 1, with balanced representation.

Given the reduced image set of ~1400 SLoF 1/2 images with annotated bounding boxes from biofouling (data set 1), we fine-tuned a YOLO binary classifier model on this data.

