## Instrumenting Salmon Movement
**Lead:** Chinook salmon were first tagged in the Gulf of Alaska
**Development:**
- Caught by hook and line
- Fish were measured and inspected for health to ensure appropriate for tagging
- Tags were applied and then fish were released back into the ocean

**Lead:** Tags then recorded data before releasing from the fish and popping up where they transmitted data over satellite
**Development:**
- Tag recording mechanisms (depth via pressure sensors, temperature sensors, temporal resolution)

**Lead:** These are then turned to positional estimates using a proprietary HMM from Wildlife computers
**Development:**
- This has underlying uncertainty so we will aggregate positions up uber h3 resolution 4

**Lead:** Only tag data collected more than one week after attachment will be retained for analysis.
**Development:**
- Fish behavior immediately following tagging may reflect surgical stress rather than natural movement
- One-week threshold allows fish to return to normal behavioral patterns
- Ensures movement data reflects natural behavior rather than tagging artifacts

**Lead:** Tag records will be screened to detect and remove data following fish mortality.
**Development:**
- Mortality indicated by cessation of depth variation in pressure measurements
- Mortality also indicated by abrupt shift to stable temperature distribution
- Removing post-mortality data prevents confounding stationary dead fish with behavioral patterns

## Classify migration versus non-migration movements

**Lead:** Expert ecologists will manually label clear examples of migration and non-migration periods to create ground truth training data.
**Development:**
- Michael Courtney and Andrew Seitz will identify unambiguous migration periods and unambiguous non-migration periods
- Labeled examples will be visualized to document what movement patterns constitute migration versus other movement types
- Clear positive and negative examples establish training foundation while classifier handles ambiguous intermediate cases

**Lead:** Features will be derived from movement data using bidirectional temporal windows around each time point.
**Development:**
- Temporal windows examine both forward and backward in time from point of interest
- Both directions needed because forward-looking alone misses migration starts and backward-looking alone misses migration ends
- Features extracted from movement data only (no environmental features)
- Feature types include momentum (sustained directional movement), behavioral consistency (erratic versus steady patterns), and temporal continuity (persistence across window)

**Lead:** A supervised classifier will be trained to predict whether each time point represents part of a migration period.
**Development:**
- Model predicts migration probability given features from bidirectional temporal window
- Model performance evaluated using accuracy on balanced dataset
- Operating point (probability threshold) selected to minimize false negative rate
- False negatives (incorrectly labeling migration as non-migration) prioritized because these would pollute non-migration training data for movement model
- Minimizing false negatives more critical than minimizing false positives for downstream model quality

**Lead:** The trained classifier will be applied to the full dataset to isolate non-migratory movements.
**Development:**
- All unlabeled time points classified using selected operating point
- Points classified as migration removed from dataset
- Resulting dataset contains only non-migration movements for movement model training
- Goal is dataset where migration movements have been filtered out with high confidence

## Preparing the Dataset

**Lead:** Stratified resampling will ensure balanced representation across space and time for unbiased model evaluation.
**Development:**
- Study area divided into spatial blocks along Alaska coast based on data distribution
- Each spatial block further divided into seasonal blocks based on data
- Samples collected into each space-time block
- Within each block, resample with replacement to prevent any individual fish from dominating
- Prevents overrepresentation of well-sampled individuals or regions in analysis

**Lead:** Comprehensive list of candidate features will be developed and categorized by type.
**Development:**
- Expert panel (Andrew Seitz, Michael Courtney, Curry Cunningham) will identify exhaustive feature list
- Features categorized as: stationary (bathymetry), temporal (temperature), or stateful (productivity/chlorophyll)
- All features standardized to Uber H3 resolution 4 and daily temporal resolution for joining with movement data
- Source datasets include Copernicus marine service covering study area and time period

**Lead:** Vector embedding analysis will cluster features sharing similar information content.
**Development:**
- Pairs of features tested for mutual predictability using single-dimension embeddings
- Features that can predict each other clustered together as possessing same information
- Iterative clustering process identifies groups where members predict each other but not members of other groups
- Representative features from each cluster become candidates for movement modeling
- Results documented in depth with supplemental table showing initial features, obtained clusters, and chosen representatives

**Lead:** Representative features from each cluster will be normalized and scaled for neural network compatibility.
**Development:**
- Features normalized to improve variability across their range
- Rescaled to [0,1] or [-1,1] depending on feature characteristics
- Prepares features for gradient-based optimization in neural networks

**Lead:** Environmental features will be joined to movement observations to create choice-labeled training dataset.
**Development:**
- Each observation includes: origin location, actual destination, and all potential destinations within 50km (or 90th percentile daily range)
- Actual destination labeled as "selected"
- Fish assigned entirely to training, validation, OR testing sets (no fish split across sets)
- Complete fish-level splitting prevents data leakage from autocorrelated movements within individuals
- Testing set held out until final model selection
- Contrast resampling employed due to large number of potential destinations relative to actual samples


## Movement Model Training

**Lead:** Series of progressively complex log-odds models will be trained to predict movement choice probabilities.
**Development:**
- MIMIC framework enables systematic exploration of feature space and hyperparameters
- Models predict likelihood of moving from origin H3 cell to each potential destination given environmental features
- Complexity increased incrementally until overfitting detected on validation set
- All models below overfitting threshold tested on held-out testing set
- Best performer on testing set selected for subsequent distribution inference

**Lead:** Individual feature importance will be assessed by training models for each feature in isolation.
**Development:**
- Features evaluated in order of their iterative contribution to full model performance
- Reveals which features are most predictive and how they influence movement decisions
- Provides mechanistic insight into environmental drivers of movement


## Bycatch Risk Model Development

**Lead:** Gulf of Alaska pollock fleet catch data will be processed and standardized for risk modeling.
**Development:**
- Bycatch binned to Uber H3 resolution 4 and daily temporal resolution
- Two standardization approaches: by effort alone, and by effort weighted by depth-occupancy likelihood
- Depth weighting uses inference results from depth distribution model
- Binary catch/no-catch dataset and normalized catch quantity dataset both created
- Datasets split into training, validation, and testing sets

**Lead:** Movement model will generate probability matrices and derive attractor indices for bycatch locations.
**Development:**
- Environmental features computed across entire Gulf of Alaska for trawl data time periods ±1 month
- Movement probability matrices predicted for same spatiotemporal extent
- For each time point, uniform initial distribution propagated forward through movement model for 1, 2, 3, and 4 weeks
- Resulting density at each cell represents "attractor index" (likelihood of fish accumulation)
- Each trawl event paired with attractor index of its location

**Lead:** Training data will consist of pairs of fishing events compared on bycatch outcomes.
**Development:**
- Pairs selected to be close in time (days apart) and space (within data density limits)
- Equal representation of: both have bycatch, only one has bycatch, neither has bycatch
- Balanced representation ensures model learns to discriminate across all relevant scenarios

**Lead:** Decision tree will learn to predict which of two locations has higher bycatch risk using only attractor-derived features.
**Development:**
- Loss function: expected bycatch given model-predicted probabilities
- Features constructed exclusively from attractor indices (no direct environmental features)
- Validation set used to explore features, model structures, and hyperparameters
- Performance reported on held-out testing set relative to 50/50 random baseline

**Lead:** Trained model will produce interpretable spatial risk maps for specific time periods.
**Development:**
- Model generates pairwise distances (difference in assigned probabilities) for all cells at interesting time points
- Agglomerative clustering groups cells with similar risk levels
- Clusters assigned colors to create risk zones (minimally green/yellow/red)
- Visualizations demonstrate spatial structure of bycatch risk as predicted from movement patterns
