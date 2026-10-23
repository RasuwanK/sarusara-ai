# Methodology

## 1. Research design

This study will design, develop, and evaluate Sarusara AI, a mobile-oriented decision-support system for identifying visible nitrogen (N), phosphorus (P), and potassium (K) deficiency symptoms in paddy leaves, estimating the proportion of leaf area affected, and translating validated agronomic rules into field-scaled fertilizer estimates. The study combines image-based classification and segmentation experiments with software development and rule-based estimator validation.

The work will use an iterative, Agile-inspired development procedure. Requirements, data, models, the image-processing pipeline, the fertilizer estimator, and the mobile interface will be developed in short cycles. Each cycle will produce a reviewable increment, which will be assessed against functional requirements and, where applicable, held-out validation data. Feedback from the supervisor, agricultural specialists, and intended users will inform the next cycle. Iteration will not replace controlled model evaluation: train, validation, and test partitions will remain separated, model selection will use training/validation data only, and the test set will be held back until the final configuration is fixed.

The computer-vision task will be staged. The first stage will classify images into the supported nutrient categories. The second stage will localize the leaf and annotate/segment the symptomatic regions. A severity index will be calculated from these masks. The fertilizer estimator will be a distinct rule-based module whose inputs, formulae, assumptions, and outputs can be tested independently of the image classifier. This separation will make it possible to identify whether an error arises from classification, segmentation, severity calculation, or recommendation logic.

## 2. Research questions and objectives

The methodology addresses the following questions:

1. How reliably can field-captured paddy leaf images distinguish the selected nutrient deficiency classes, including healthy leaves?
2. Which image-processing pipeline and deep-learning architecture provide the best class-wise performance and practical mobile inference characteristics?
3. How accurately can leaf and symptom regions be segmented, and how closely does a pixel-area severity index agree with expert reference annotations?
4. Can government agronomic recommendations be encoded in a fertilizer estimator that scales nutrient/fertilizer quantities to the user's field area and crop context?
5. Does iterative integration of these components produce a usable mobile decision-support prototype whose outputs can be traced and evaluated?

The project objectives are to construct a locally relevant dataset; compare multiple deep-learning models; develop image processing and symptom localization; calculate and validate severity; build and test an area-based fertilizer estimator; and evaluate the integrated mobile system.

## 3. Iterative development process

Development will follow short, Agile-inspired iterations with a defined objective, implementation, review, and backlog update. Iterations will cover: (i) requirements and initial dataset audit; (ii) image capture, labeling, and data management; (iii) baseline preprocessing and classifier; (iv) architecture and preprocessing comparisons; (v) leaf/symptom segmentation and severity index; (vi) fertilizer-rule encoding and estimator tests; and (vii) mobile integration and end-to-end evaluation. At the end of each iteration, deliverables will be reviewed with the supervisor and, where available, an agronomy advisor. Defects, data gaps, and performance failures will be recorded and prioritized for the next iteration.

The process is iterative in implementation, but the research comparisons will remain reproducible. A version-controlled experiment log will record dataset version, group split, preprocessing choices, model architecture, hyperparameters, random seed, software versions, checkpoint, and evaluation results. Once final model and thresholds are selected using validation results, the held-out test data will be used for final evaluation without further tuning.

## 4. Dataset collection and ground truth

### 4.1 Local paddy leaf collection

The study will include locally collected paddy leaves so the dataset reflects Sri Lankan varieties, field conditions, cultivation practices, and image-capture environments. Collection will include symptomatic leaves representing the target N, P, and K categories and healthy leaves as a reference/control category. Healthy examples will be collected from local fields and, where feasible, matched to symptomatic examples by site, cultivar, growth stage, date/season, and capture conditions. This matching will reduce the risk that a model learns location, background, or camera differences instead of plant symptoms.

Images will be captured with standard smartphones under varied but documented field conditions. The capture protocol will request a clear view of the whole leaf, with optional close-up images for symptom details, and will record image ID, pseudonymous plant/plot ID, site or district at a non-identifying level, date/season, cultivar if known, growth stage, device, lighting, and capture notes. Original images will be retained unchanged. Repeated views of the same leaf or plant will be linked by group ID so they can be kept within one data partition. Consent and permissions will be obtained as required, and identifying farmer information will not be stored with image labels.

The proposal's target of at least 1,500 images will guide collection, but image count alone will not be treated as sufficient evidence. The study will report class counts and the number of independent plants, plots, sites, seasons, and devices. Images will be quality-checked for blur, occlusion, duplicates, and incorrect labels. Exclusions and reasons will be logged. If classes remain imbalanced, class-aware sampling or training loss weighting may be used on training data only.

### 4.2 Government reference data and agronomic sources

Government data will provide the local agronomic reference inputs used to define severity interpretation and fertilizer-estimator rules. The Department of Agriculture (DOA), including the Rice Research and Development Institute, publishes paddy fertilizer recommendations by cultivation context, soil-analysis results, and leaf-colour index; it also provides divisional fertilizer recommendations based on soil information [1, 2]. These documents and tables will be reviewed and recorded with their issuing body, publication/version date, applicable zone, irrigation/rainfed context, growth stage, units, and source URL. Paddy extent, yield, and production data from the Department of Census and Statistics may be used to describe the study context, not as image-level symptom labels [3].

The government sources do not provide the pixel masks required to compute the image-based severity index. Accordingly, government records will inform agronomic interpretation, reference severity categories where an applicable official scale exists, and fertilizer rules; image-level severity values will be generated from this study's locally collected and expert-annotated images. If no official severity scale is available for a nutrient symptom, the study will not attribute an invented threshold to a government agency. It will define and validate any study-specific ordinal grade with agricultural specialists and report it as a study protocol.

### 4.3 Labeling and annotation protocol

A label handbook will define the visual and agronomic criteria for each class. Initial labels will include healthy/no visible target deficiency and N, P, and K deficiency. Where field images may contain other diseases, other nutrient problems, mixed deficiencies, or uncertain symptoms, annotators will record these states rather than force them into one of the three target classes. Diagnosis evidence will also be recorded: visual assessment, agronomist review, controlled treatment, soil/tissue analysis, or other available evidence. Visual labels alone will be identified as such and will not be presented as laboratory-confirmed nutrient status.

Two trained annotators will independently review a subset of images, including difficult and mixed cases. Disagreements will be adjudicated with an agricultural specialist. Inter-rater agreement will be reported using an appropriate statistic (for example, Cohen's kappa for categorical labels). Low-confidence cases will remain flagged, and the analysis will state how they are handled.

For severity estimation, annotators will create two pixel-level masks for selected images: (a) the visible leaf area and (b) the target nutrient-symptom areas on that leaf. The diseased/deficient areas of each annotated leaf must be marked separately from the healthy leaf tissue. Where multiple deficiency symptoms occur, separate masks will be created by nutrient class if the evidence permits. Annotation instructions will define leaf boundaries, yellowing/discoloration transitions, holes, shadows, occlusion, overlapping leaves, and uncertain pixels. A representative subset will be independently annotated by two raters and reconciled to assess annotation agreement.

### 4.4 Data partitioning

Images will be divided into approximately 70% training, 15% validation, and 15% test data using stratified group splitting. Plant, plot, or capture-session group IDs will be used as grouping units so related images cannot appear in both training and test sets. Where the number and distribution of sites permit, a later collection period or site will be reserved as an additional external test. Exact and near-duplicate images will be identified before partitioning. The split manifest and random seed will be saved. Preprocessing thresholds and model selection will use the training and validation partitions only.

## 5. Image preprocessing and feature extraction

Images will be processed through a consistent pipeline while preserving the original files. The baseline pipeline will decode to RGB, correct orientation, preserve aspect ratio, resize or pad to model input dimensions, and apply the normalization required by the selected architecture. Image augmentation will be used only during training and will include plausible rotations, flips, scale/crop variation, and modest brightness/contrast/color changes. Augmentation will not alter symptom appearance unrealistically. Validation and test transformations will be deterministic.

Image-processing effects will be assessed by controlled ablation against the baseline rather than applying every operation together without measurement. Candidate operations include Gray World or other white-balance correction; conversion to CIELAB or HSV to separate brightness and chromatic cues; CLAHE on the luminance channel; Excess Green/color-based masking; and clustering for initial leaf-background separation. Denoising will be included only if image noise is observed to interfere with recognition. Each pipeline will be applied to identical data partitions so changes in performance can be attributed to the tested operation.

For classification, the image tensor and learned convolutional features will be used by the model. For segmentation and severity, interpretable image features will additionally be extracted: color-channel values or distributions, local contrast, texture, connected-region size and shape, and pixel counts. These hand-engineered features can serve as a thresholding baseline and error-analysis aid. They will not replace expert masks as the reference target.

## 6. Deep-learning model development and comparison

Several deep-learning models will be trained and evaluated against one another to select the best-performing approach for the target use case. The existing ResNet-50 training script will provide a reproducible baseline. Candidate models will include a lightweight mobile architecture such as MobileNet, a further transfer-learning model such as EfficientNet, and a small CNN baseline if computation and dataset size allow. The same group-aware partitions, input data, and comparable training budgets will be used for fair comparison. More than one random seed will be used for finalists where feasible.

For a single-label task, models will output a softmax distribution across healthy, N, P, K, and any other supported class. If adjudicated examples can have multiple simultaneous nutrient deficiencies, a multi-label formulation with independent sigmoid outputs will be used for those labels; evaluation and threshold selection will follow the multi-label protocol. Early stopping and checkpoint selection will use validation data, not test performance. Class weighting will be determined from training-set counts only.

Classification will be reported using accuracy, balanced accuracy, per-class precision/recall/F1, macro-F1, confusion matrix, one-vs-rest precision-recall curves and average precision. Confidence intervals will be calculated with resampling at the plant/plot group level where possible. Error analysis will examine class, site, growth stage, lighting, device, and background. For mobile suitability, model size and inference latency will be measured on named target devices under documented conditions. A high-performing model will not be selected on accuracy alone if it is too large or slow for the intended application.

## 7. Leaf and symptom localization and severity index

### 7.1 Mask extraction from image features

Severity will be calculated from leaf and symptom masks, not from classification confidence or bounding-box size. First, a leaf mask will be generated. Color or Excess Green thresholding and clustering will provide an interpretable baseline for separating green leaf pixels from soil, water, and other background. If field backgrounds, shadows, or overlapping leaves make this baseline unreliable, a semantic-segmentation model will be trained from manually annotated leaf masks. The mask will be checked for disconnected background components, gaps, and excluded occluded portions.

Second, a symptom mask will be generated only within the leaf mask. Candidate symptoms will be represented through color features in CIELAB/HSV, local contrast, and texture; these support a simple thresholding baseline. The primary learned approach will use a semantic-segmentation model trained on expert-annotated nutrient symptom pixels. Thresholds for color rules or model probabilities will be selected on training/validation data and frozen before final test evaluation. If multiple nutrient classes occur, separate symptom masks will be evaluated for N, P, and K where annotator evidence supports them.

The primary image features for severity calculation will be the number of valid leaf pixels and the number of target symptom pixels inside the leaf mask. Additional exploratory features may include symptomatic color proportion, connected symptom-region count, region size distribution, boundary/shape descriptors, and texture features such as gray-level co-occurrence matrix (GLCM) contrast, energy, and homogeneity. These features may help characterize symptom patterns or predict expert-assigned grades, but the continuous severity index will remain a masked area ratio.

### 7.2 Severity index calculation

For one leaf image, the severity index (SI) will be calculated as:

**SI (%) = 100 x (number of target-symptom pixels inside the valid leaf mask / number of valid visible leaf pixels).**

The numerator will include only pixels labeled as the target nutrient symptom and located within the leaf mask. The denominator will include visible leaf pixels eligible for analysis. Background, severe occlusion, and unusable pixels will be excluded using annotation rules specified before evaluation. SI will be calculated for each leaf image. If several leaves are sampled from one plant, the study will report individual leaf scores and a plant-level, area-weighted aggregate using a pre-defined rule. A separate N, P, or K SI will be reported only when the corresponding symptom class can be distinguished and annotated; mixed or uncertain areas will be separately flagged.

Government agronomic material will be used as the reference source for applicable leaf-colour index guidance, fertilizer recommendation context, and any officially defined severity categories. It will not be represented as having supplied pixel-level SI values. Where official material provides categorical thresholds, those thresholds will be transcribed with source/version and checked with an agricultural specialist. Where there are no applicable official thresholds, any ordinal severity bands will be developed as study-specific thresholds and validated against expert assessment rather than attributed to government data.

### 7.3 Severity validation

Predicted leaf and symptom masks will be compared with expert masks using Dice coefficient and Intersection over Union (IoU), with visual boundary review. The continuous SI will be compared with expert area estimates using mean absolute error (MAE), root mean squared error (RMSE), and an agreement analysis. If ordinal grades are used, their thresholds will be fixed before test evaluation and evaluated using a confusion matrix and class-wise agreement. Grad-CAM and related saliency maps may be used to inspect model attention but will not be treated as segmentation masks or as measured symptom area.

## 8. Fertilizer estimator development and testing

The fertilizer estimator will be developed as a separate, transparent module. It will translate an applicable agronomic nutrient recommendation into estimated quantities of available commercial fertilizers and scale those quantities to the user's field area. The estimator will use government-issued DOA paddy fertilizer recommendations as the primary ruleset, including differences by agro-ecological zone, irrigation/rainfed system, soil-analysis result or available divisional baseline, crop stage, and leaf-colour index where applicable [1, 2]. Each rule will retain the source, effective date/version, units, conditions, and assumptions. A qualified agricultural specialist will review the encoded rules before testing.

The estimator inputs will include field area and unit, location/zone or GN division where relevant, cultivation system, crop age/stage, and available soil or leaf-colour information. If a soil-test value or required agronomic input is absent, the estimator will identify the missing value and use only an explicitly approved default or display that an estimate cannot be produced. A nutrient recommendation will be converted into product mass using the declared nutrient concentration of each fertilizer product. The calculation will be dimensionally checked and scaled by field area; the output will display both the per-unit-area recommendation and total field quantity, with units, application timing/splits, and the source rule. The implementation will use the selected commercial products (for example Urea, DAP/TSP, and MOP) only where their nutrient composition is appropriate to the referenced recommendation. Substitution and fertilizer grades will be confirmed against the current DOA guidance rather than assumed.

The estimator will be tested using a set of hand-calculated cases spanning different field sizes, zones, water regimes, soil/leaf-colour inputs, and crop stages. Unit tests will check area conversion, product nutrient fraction, split application totals, missing inputs, boundary values, and rounding. Property checks will confirm that, for the same agronomic recommendation, doubling field area doubles the estimated mass and that unit conversions produce equivalent results. Output quantities will be compared with the relevant DOA tables or worked examples and reviewed by an agricultural specialist. If a rule calls for a rate R kg/ha and the field area is A ha, the field nutrient or product amount will be calculated as R x A after applying the documented nutrient-to-product conversion factor. Recommendations will not be inferred directly from image SI alone unless a scientifically and agronomically validated mapping from SI to nutrient dose is established. The prototype will show the visual finding and fertilizer estimate as linked but distinct outputs, with uncertainty and source details.

## 9. Integrated mobile system and evaluation

The mobile prototype will allow a user to capture or select a paddy leaf image, review the image-quality prompt, receive the classifier result and confidence, view localized symptom regions and SI when validated, enter field/crop inputs, and view an area-scaled fertilizer estimate with its source and assumptions. Low-confidence, out-of-scope, mixed, or poor-quality images will be flagged for retake or field/agronomic assessment instead of yielding unsupported certainty. The system will record only the information needed for evaluation and will protect farmer identity and precise location.

System evaluation will be separated into component-level and end-to-end measures. Component evaluation will cover classification, segmentation, SI agreement, and fertilizer arithmetic/reference correctness. End-to-end evaluation will assess whether the integrated flow produces the expected result for pre-defined scenarios, handles invalid or missing input safely, and presents units and recommendation context clearly. If user evaluation is feasible, a small usability study with representative users or agricultural personnel will assess task completion, comprehension, and usability; it will be reported separately from model accuracy.

## 10. Reproducibility, validity, and ethics

The project will maintain a data dictionary, image and annotation protocol, exclusion log, grouped split manifest, source register for official agronomic data, model configurations, experiment logs, and estimator test cases. The final test data will remain untouched until the pipelines and thresholds are fixed. Results will report class balance, number of independent collection groups, missing or uncertain labels, site/device coverage, and deviations from protocol.

Potential limitations include visual similarity between nutrient deficiency and biotic disease, limited local data, uneven class/site representation, symptom variation by cultivar and growth stage, and changes in official fertilizer guidance. The system will therefore be framed as decision support rather than a replacement for soil testing or qualified agronomic advice. Personal data will be minimized, image collection permissions will be respected, and recommendations will not be presented as definitive where required agronomic inputs are missing.

## References

[1] Department of Agriculture, Sri Lanka, Rice Research and Development Institute. “Rice in Sri Lanka: Fertilizer Recommendations.” https://doa.gov.lk/rrdi_fertilizerrecomendation/ (accessed 22 October 2026).

[2] Department of Agriculture, Sri Lanka. “Fertilizer Recommendation for Paddy Cultivation in Sri Lanka: Grama Niladhari Division Level.” https://doa.gov.lk/fertilizer-recommendation/ (accessed 22 October 2026).

[3] Department of Census and Statistics, Sri Lanka. “Paddy Statistics.” https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics (accessed 22 October 2026).
