# Sarusara AI 🌾

**Sarusara AI** is an intelligent agricultural mobile application designed to help farmers optimize crop yields by identifying nutrient deficiencies in paddy leaves. By leveraging edge-based computer vision and environmental data fusion, the system provides precise, real-time, and localized fertilizer recommendations to prevent crop stress and minimize chemical waste.

### Core Features

* **Automated Deficiency Detection:** Utilizes a mobile-optimized YOLO object detection model to accurately identify Nitrogen (N), Phosphorus (P), and Potassium (K) deficiencies simultaneously under varying field conditions.
* **Dynamic Severity Indexing:** Calculates the exact severity of leaf lesions using localized contrast enhancement, illumination adjustment, and semantic masking techniques.
* **Context-Aware Agronomic Fusion:** Cross-references visual stress data with crop phenology (age and growth stage), estimated soil baselines, and live weather telemetry to adjust recommendations dynamically (e.g., delaying application if heavy rain is forecasted).
* **Precision Fertilizer Mapping:** Translates raw nutrient deficits into actionable, commercially available straight fertilizer dosages (Urea, DAP, and MOP) scaled directly to the user's specific field dimensions.

### ResNet-50 nutrient classifier

The training script `train_resnet50.py` classifies a paddy leaf image as nitrogen,
phosphorous, or potassium deficient using the corresponding folders in
`dataset/images/`. Install project dependencies with `pip install -r requirements.txt`,
then run `python train_resnet50.py`. The default run fine-tunes ImageNet-pretrained
ResNet-50 for 20 epochs. It uses CUDA and mixed precision automatically when a
compatible GPU is available; on Windows, the default data-loader worker count is 0.

The script makes stratified 70/15/15 train, validation, and test splits. It saves
the best validation-loss checkpoint, test metrics, a confusion matrix, and
one-vs-rest precision-recall curves with training history under
`outputs/resnet50/`. Useful options include `--epochs`, `--batch-size`,
`--freeze-backbone` (faster head-only fine-tuning), `--workers`, and `--no-pretrained`
(avoid the ImageNet weight download).

### System Architecture Highlights

* **Pre-processing Pipeline:** Standardizes raw smartphone images by mitigating sun glare, removing soil/weed backgrounds, and normalizing color spaces.
* **Detection Engine:** YOLO-driven bounding box evaluation against a total leaf pixel mask.
* **Decision Matrix:** Multi-dimensional rule-based logic integrating external environmental APIs.
* **Recommendation Engine:** Outputs localized, mass-based dosage plans designed to prevent fertilizer leaching.
