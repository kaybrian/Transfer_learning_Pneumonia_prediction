# Transfer Learning Assignment: Pneumonia Detection from Chest X-Ray Images

## Problem Statement
The objective of this project is to leverage transfer learning to detect pneumonia from chest X-ray images. Pneumonia is a lung infection that can be bacterial or viral, leading to severe respiratory issues if untreated. Early detection is crucial for effective treatment, and deep learning models can assist in automating this diagnostic process, especially in resource-limited settings.

## Dataset
We used a dataset of chest X-ray images sourced from pediatric patients (1 to 5 years old) at the Guangzhou Women and Children’s Medical Center. The dataset is organized into three main folders: `train`, `test`, and `val`, each containing subfolders for two categories: `Pneumonia` and `Normal`. The images were screened for quality, graded by expert physicians, and evaluated by an additional expert to minimize errors.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.


## Dataset Link: 
- Data: [Chest X-Ray Pneumonia Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

## Pre-trained Models Used

We selected the following pre-trained models based on their architecture and suitability for image classification tasks involving fine-grained textures, such as distinguishing between healthy and diseased lung tissues:

- VGG16: Known for its simplicity and effectiveness in capturing fine details in images.
- InceptionV3: Recognized for its ability to handle complex image classification tasks with efficient feature extraction.
- EfficientNetB0: Offers state-of-the-art accuracy with fewer parameters, making it efficient for medical image analysis.


## Justification for Model Selection

- VGG16: Its deep architecture with multiple convolutional layers makes it suitable for capturing texture details in X-ray images, which is crucial for differentiating between pneumonia and normal lung images.
- InceptionV3: It balances complexity and performance by using a modular approach to capture different scales of image features, enhancing its ability to classify varied patterns seen in medical imaging.
- EfficientNetB0: It uses compound scaling to optimize depth, width, and resolution, making it highly effective for medical image analysis with relatively lower computational cost.


## Fine-Tuning Process
For each model, the pre-trained weights from ImageNet were used as a starting point. We replaced the top layers of the models with new layers tailored to our binary classification task:

- Global Average Pooling Layer: To reduce dimensions and avoid overfitting.
- Dense Layers: To introduce complexity specific to pneumonia detection.
- Dropout Layers: To prevent overfitting by randomly disabling neurons during training.

The modified models were trained on the augmented dataset with a learning rate fine-tuned for each architecture. The last few layers were unfrozen and retrained to adapt the feature extraction capabilities of each model specifically to our dataset.


## Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy: Measures the overall correctness of predictions.
- Loss: Binary cross-entropy loss used for training and evaluation.
- Precision: Indicates the proportion of true positive predictions among all positive predictions.
- Recall: Measures the ability of the model to detect all relevant cases (pneumonia).
- F1 Score: Harmonic mean of precision and recall, providing a balanced measure of performance.

## Results


| Model        | Accuracy | Loss  | Precision | Recall | F1 Score |
|--------------|----------|-------|-----------|--------|----------|
| **VGG16**    | 88.78%   | 0.3345 | 53.00%    | 55.00% | 54.00%   |
| **InceptionV3** | 83.81% | 0.3812 | 53.00%    | 56.00% | 54.00%   |
| **EfficientNetB0** | 62.50% | 0.6897 | 39.00%    | 62.00% | 48.00%   |


## Discussion

The performance of the models on the task of pneumonia detection from chest X-ray images shows varying strengths and weaknesses, highlighting the impact of different architectures on medical image analysis.

### 1. VGG16

- Strengths: VGG16 achieved a high accuracy of 88.78% with relatively good balance between precision (53%) and recall (55%). The model's ability to capture fine-grained textures helped in distinguishing between normal and pneumonia cases.
- Weaknesses: The model's recall rate indicates that while it performs reasonably well in detecting pneumonia, it still misses a notable number of true cases, reflecting a need for further fine-tuning to improve sensitivity.

### InceptionV3

- Strengths: InceptionV3 provided a similar accuracy of 83.81% with a slightly better recall (56%) compared to VGG16. Its modular approach allowed it to capture features across different scales, contributing to its balanced performance in classifying X-ray images.
- Weaknesses: Precision was relatively low at 53%, indicating a significant number of false positives. This suggests that while the model is good at capturing actual pneumonia cases, it also incorrectly identifies healthy cases as pneumonia, which could lead to unnecessary further investigations in a clinical setting.


### EfficientNetB0

- Strengths: EfficientNetB0 showcased high recall (62%), demonstrating its strong ability to detect pneumonia cases, which is critical in medical diagnostics where false negatives can be detrimental. Its efficient architecture also makes it suitable for deployment in resource-constrained environments.
- Weaknesses: The model had the lowest precision (39%) and overall accuracy (62.50%) among the tested models. This high rate of false positives indicates that while EfficientNetB0 is good at identifying most pneumonia cases, it struggles significantly with correctly identifying normal cases, which could lead to over-diagnosis.


## Summary of Findings

- Overall Performance: VGG16 and InceptionV3 provided a balanced performance with a good trade-off between accuracy and sensitivity. EfficientNetB0, while efficient in terms of resource usage and sensitivity, requires further tuning to improve its precision.
- Transfer Learning Strengths: All models benefited from transfer learning, allowing them to quickly adapt to the medical domain with minimal training time and achieve reasonable performance on the chest X-ray dataset.
- Transfer Learning Limitations: The models are sensitive to class imbalance and noise in the data, as seen from the varied precision and recall metrics. Fine-tuning transfer learning models requires careful adjustment of layers and parameters to align them with the specific medical imaging task.


## Conclusion

Transfer learning proved highly effective for pneumonia detection from chest X-rays, with InceptionV3 providing the best overall performance. This approach can be extended to other medical imaging tasks to enhance diagnostic accuracy and speed in clinical practice


## Authors 
- Kayongo Johnson Brian - b.kayongo@alustudent.com