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

1. Biofouling dataset, a collection of ~10,000 underwater images labeled with simplified level of fouling (SLoF) 0, 1, or 2, also described as nil, medium, and heavy fouling.
This data set is described in [Automating the assessment of biofouling in images using expert agreement as a gold standard](https://www.nature.com/articles/s41598-021-81011-2), 
and some example images are available at [GitHub](https://github.com/emannix/automating-the-assessment-of-biofouling/tree/main). The data is heavily class imbalanced, with over 7000 entries in class 0.

2. Fouling detection dataset of ~800 images with bounding box annotations formatted for YOLO, available on [Roboflow](https://universe.roboflow.com/abdul-vdimo/fouling-detection-m3h28). 
Each bounding box is labelled with one of 7 classes such as clean, rust, starfish, barnacles, etc.

## Baseline random forest classifier

We split all 10,000 images in data set 1 into 80/10/10 train/test/validation, and trained two baseline SLoF classification models: a dummy classifier and random forest classifier. 
Because of class imbalance, the dummy classifier has relatively high accuracy (0.77) but is terrible on every other metric.

The random forest classifier utilizes features vessel.id, niche.area, and paint.quality, as well as some complexity-related engineered features: 
Shannon entropy, gradient energy, Laplacian entropy, and variances of each red, green, and blue channels. 
The random forest classifier does not directly process image files; it is only based upon these numerical features computed from the image.
The random forest was an ensemble of 500 trees with max depth 10, and balanced class weighting.
As expected, this random forest classifier is very imprecise. 
Surprisingly, the highest feature as ranked by feature importance score was the paint.quality feature present in the original data set.

## ResNet and EfficientNet

We experimented with ResNet18 and EfficientNet based models after implementing custom class weights to counter for class imbalance, 
and found that ResNet18 has better performance in classifying between all 3 SLoF levels. 
For SLoF level 0, 1 and 2, the obtained precision was 0.851, 0,366, 0.456, with recall 0.668, 0.409, 0.701 respectively, 
with a validation macro-F1 score over all 3 classes as 0.5623. As ResNet18 was trained trained from scratch 
(without ImageNet pretraining) we could tell from the confusion matrix that the model did learn fouling feature gradient 
and can successfully differentiate between nil and heavy fouling (confusing levels 0 and 2 is less than 0.05), 
but medium fouling category is very prone to be misclassified as either nil (~0.2) or heavy (~0.3). 
EfficientNet also did a better job in disguising between nil and heavy, but it misclassified almost all medium fouling images.

Hence, we made a decision to merge medium and heavy category together as one "biofouling" category, and moved to YOLOv8m-cls model, 
that is better than both ResNet18 and EfficientNet in binary classification. The fouled category had a precion of 0.74, 
recall 0.65 and f1 0.69, where the nil category had a precision of 0.89, recall 0.93, and f1 0.91, 
with the overall accuracy of 0.86, AUC 0.868, macro F1 0.80, which is a strong result for a binary classifier on imbalanced data. 

## Data augmentation and cleaning, SAM3

To improve further, it was necessary to detect and create bounding boxes for fouling regions, 
as well as estimate percentage of surface area for biofouling contamination from underwater images. 
Hence, we turned to YOLOv8s model, a powerful pre-trained computer vision model with 11 million parameters, trained on ImageNet backbone. 

In order to use YOLO, we needed data in a particular format, specifically data including certain kinds of bounding boxes.
Data set 2 already has these bounding boxes, but data set 1 does not. In order to take advantage of the larger size of data set 1,
we developed a pipeline for processing images in data set 1 to add bounding boxes, utilizing SAM3 (Segment Anything Model). 

First, we sorted images in data set 1 into SLoF level 0 (~7000) and SLoF levels 1 and 2 (~3000).
For level 0 (little or no fouling), we simply set a single bounding box for the entire image.
For levels 1 and 2 (medium/high fouling), we first manually placed bounding boxes on around 50 images, then
with a combination of mimicry and text prompts instructed SAM3 to place bounding boxes in the remaining images.
Due to various technical limitations, this only worked for ~1400 images, so we end up with ~1400 images 
of SLoF level 2 or 3 as a subset of  data set 1 usable with YOLO.

## YOLO

We fine-tuned the YOLOv8s model on data set 2 to label each bounding box as one of 7 classes (clean surface, starfish, barnacle, etc.). 
Then we merged the 6 non-clean classes as "biofouling", and kept the "nil" fouling, to be compatible with our binary classifier. 
The idea is to investigate how the model can learn biofouling from a completely different set of data, and perform on dataset 1.

Then we pursued two directions:

A. Apply that YOLO model to the annotated data set 1 as a validation set.
B. Retrain YOLO model on full annotated data set 1, with balanced representation.

Given the reduced image set of ~1400 SLoF 1/2 images with annotated bounding boxes from biofouling (data set 1), 
we implemented class weights and then fine-tuned the YOLOv8s binary classifier model, and ran on the validation set. 
Option A almost did not pick up anything from dataset 1, whereas Option B returned with decent classification, 
with nil precision 0.9050, recall 0.9898 and f1 0.9455 and biofouling precision 0.5473, recall 0.2020, and f1 0.2951, with IOU set to 0.5.

