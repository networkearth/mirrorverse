## Validated Tag Movement Data
**Lead:** Acquire pop-up satellite tag data from Chinook salmon in the Gulf of Alaska
**Development:**
- Existing tag deployment data from collaborative research efforts will be compiled for this analysis
- Deployments span multiple locations: Dutch Harbor, AK (n=20), Chignik, AK (n=16), Craig, AK (n=8), Homer, AK (n=20), Kodiak, AK (n=13), Yakutat, AK (n=16), Sitka, AK (n=15), and Central Bering Sea (n=3)
- Fish captured by hook and line at multiple locations across the Gulf of Alaska
- Only healthy individuals meeting minimum size criteria (62-100cm fork length) selected for tagging
- Pop-up satellite archival tags attached following established protocols
- Tagged fish released at capture location
- Tagging conducted across spatial distribution of the population to ensure representative coverage
- Pop-up satellite archival tags on free-ranging individuals provide the spatiotemporal movement trajectories necessary to train movement probability models

**Lead:** Collect time-series depth, temperature, and position data from surfaced tags
**Development:**
- Tags collect temperature, ambient light intensity, and depth information at specified (sub/day) intervals during deployment - these measurements enable light-based geolocation with temperature-based corrections using surface temperature data
- Tags programmed to release from fish, surface, and transmit collected data over satellite
- Data transmitted in time-series format with measurements at 2.5-15 minute intervals before uploading
- Satellite transmission is constrained by battery life, but in the case of physical tag recovery, complete high-resolution data can be retrieved without compression losses from satellite transmission

**Lead:** Convert raw tag transmissions to spatially-gridded position estimates at H3 resolution 4
**Development:**
- To obtain location estimates, the data from the tag is passed through a proprietary hidden Markov model from Wildlife Computers that uses the tag’s data to estimate the likely paths taken by the fish during tag deployment (Wildlife Computers, 2025). 
- An estimate of location for each day can then be derived as the central tendency of those paths on a particular day. 
- Position estimates converted to Uber H3 hierarchical hexagonal grid at resolution 4 (~26km edge length hexagons)
- Aggregation to H3 resolution 4 accommodates geolocation uncertainty while maintaining meaningful spatial resolution
- Daily positions assigned to H3 cell IDs creating spatially-standardized movement tracks

**Lead:** Filter tag records to retain only post-acclimation movement data
**Development:**
- Fish behavior immediately following tagging may reflect surgical stress rather than natural movement
- All movement data from first seven days post-deployment excluded from analysis
- One-week threshold allows fish to return to normal behavioral patterns based on previous tagging studies
- Filtering applied to each individual fish based on their specific deployment date
- Ensures movement data reflects natural behavior rather than tagging artifacts

**Lead:** Screen and remove post-mortality records from the movement dataset
**Development:**
- Constant depth readings (no variation in pressure measurements) indicate mortality with the tag either on the seafloor or at the surface
- Temperature stabilizing at internal body temperature of known salmon predators, differing from ambient water temperature, indicates predation mortality
- Michael Courtney and Andrew Seitz will identify which predator body temperatures to use based on their previous Chinook salmon mortality research
- Each fish's track scanned to identify mortality transition point based on these indicators
- All data following detected mortality excluded from dataset
- Removing post-mortality data prevents confounding stationary dead fish with behavioral patterns

**Lead:** Visualize validated trajectories and summarize data retention across filtering steps
**Development:**
- Map showing all validated movement trajectories to demonstrate geospatial coverage across Gulf of Alaska
- Summary table reporting data retention: total tags deployed, tags recovered, data points before/after acclimation filter, data points before/after mortality filter
- Individual-level statistics showing distribution of track durations and data point counts per fish
- Demonstrates spatial extent of data and quantifies impact of quality control steps

## Migration Classification Model

**Lead:** Manually label migration and non-migration periods to create ground truth training data
**Development:**
- Michael Courtney and Andrew Seitz will identify unambiguous migration periods and unambiguous non-migration periods through manual expert labeling, as no standard methodology exists for this classification
- Clear positive and negative examples establish training foundation while classifier handles ambiguous intermediate cases

**Lead:** Derive temporal movement features using bidirectional time windows
**Development:**
- Temporal windows examine both forward and backward in time from point of interest
- Both directions needed because forward-looking alone misses migration starts and backward-looking alone misses migration ends
- Window sizes constrained to one week or less to retain data from fish with shorter tagging periods
- Features extracted from movement data only (no environmental features), as migration should be identifiable from the nature of the movement alone
- Feature types include momentum (sustained directional movement), behavioral consistency (erratic versus steady patterns), and temporal continuity (persistence across window)

**Lead:** Train a supervised classifier to predict migration periods from movement features
**Development:**
- Candidate models include random forests, XGBoost, and neural networks (other architectures considered if issues arise)
- Class balancing applied to ensure equal representation of migration and non-migration periods during training
- Model predicts migration probability given features from bidirectional temporal window
- Model performance evaluated using accuracy on balanced validation dataset
- Operating point (probability threshold) selected to achieve false negative rate below 5% of overall dataset
- False negatives (incorrectly labeling migration as non-migration) prioritized because these would pollute non-migration training data for movement model
- Minimizing false negatives more critical than minimizing false positives for downstream model quality

**Lead:** Evaluate classifier performance and quantify data filtering impact
**Development:**
- Confusion matrix showing true/false positives and negatives on validation set
- ROC curves for both training and testing sets to assess generalization and identify overfitting
- Summary table reporting data filtered: total time points, points classified as migration, points classified as non-migration
- Individual-level statistics showing distribution of migration periods per fish
- Demonstrates classifier performance and quantifies filtering impact

**Lead:** Visualize example trajectories with classified migration and non-migration segments
**Development:**
- Map examples showing trajectories with migration segments (red) and non-migration segments (blue) overlaid
- Examples selected to illustrate range of movement patterns observed in classified data
- Demonstrates classifier successfully distinguishes migration patterns in geospatial context

**Lead:** Apply the trained classifier to isolate non-migratory movements from the full dataset
**Development:**
- All unlabeled time points classified using selected operating point
- Points classified as migration removed from dataset
- Resulting dataset contains only non-migration movements for movement model training
- Goal is dataset where migration movements have been filtered out with high confidence

## Feature-Enriched Training Data

**Lead:** Resample movement data to ensure balanced spatial and temporal representation
**Development:**
- Study area divided into four spatial regions: Aleutian Island Chain, Kodiak to Yakutat, Southeast Alaska (to British Columbia border), and British Columbia and south
- Spatial regions correspond with typical fisheries management boundaries in the Gulf of Alaska
- Each spatial region further divided into seasonal blocks (Winter, Spring, Summer, Fall) as Chinook behavior varies seasonally (Gietzmann-Sanders, in review; Seitz, 2024) and finer temporal resolution would spread data too thinly
- Samples collected into each space-time block with equal representation across all blocks
- Within each block, resample with replacement to prevent any individual fish from dominating (target sample size per block determined from data distribution)
- Prevents overrepresentation of well-sampled individuals or regions in analysis, ensuring the model is not biased toward any one region or season when making predictions across the entire Gulf of Alaska

**Lead:** Develop and categorize comprehensive list of candidate environmental features
**Development:**
- Exhaustive feature list compiled from published literature and expert panel input (Andrew Seitz, Michael Courtney, Curry Cunningham) to ensure consideration of features beyond those typically used
- Exhaustive initial list ensures feature selection is driven by information criteria rather than human opinion, allowing data-driven prioritization in subsequent clustering steps
- Features categorized as: stationary (bathymetry), temporal (temperature), or stateful (productivity/chlorophyll)
- All features standardized to Uber H3 resolution 4 and daily temporal resolution for joining with movement data
- Source datasets include Copernicus marine service covering study area and time period

**Lead:** Cluster features by information content using vector embedding analysis
**Development:**
- Pairs of features tested for mutual predictability using single-dimension neural network embeddings, where successful single-dimension embedding indicates feature redundancy
- Neural network embeddings capture nonlinear relationships between features, ensuring redundancy detection isn't limited to linear correlations as features will be used in nonlinear neural network models
- Predictability measured by how well the embedded feature predicts the input features
- Features that can predict each other clustered together as possessing same information
- Clustering identifies groups where members can be collectively embedded into one or a few features
- Representative feature selected from each cluster as the most predictive of other cluster members
- Single representative per cluster chosen for interpretability, though feature embeddings themselves could be used if representative features prove insufficient
- Representative features from each cluster become candidates for movement modeling

**Lead:** Summarize feature clustering results showing groupings and selected representatives
**Development:**
- Supplemental table documenting: initial features, obtained clusters, and chosen representative from each cluster
- Summary statistics showing number of features per cluster and cluster characteristics (stationary vs temporal vs stateful)
- Demonstrates which features share information and justifies representative selection

**Lead:** Normalize and scale representative features for neural network compatibility
**Development:**
- Features normalized to improve variability across their range
- Features rescaled to [0,1] if non-negative (representing scales or magnitudes) or [-1,1] if containing negative values, keeping all features in similar ranges to ensure similar gradient magnitudes across features during optimization
- Prepares features for gradient-based optimization in neural networks

**Lead:** Join environmental features to movement observations to create choice-labeled training dataset
**Development:**
- Each observation includes: origin location, actual destination, and all potential destinations within the 90th percentile daily range to capture realistic movement options
- Actual destination labeled as "selected"
- Choice-based formulation enables prediction of movement probabilities to each potential destination, which are used to construct movement probability matrices for distribution inference
- Fish assigned entirely to training, validation, OR testing sets (no fish split across sets)
- Complete fish-level splitting prevents data leakage from autocorrelated movements within individuals
- Testing set held out until final model selection
- Contrast resampling employed due to large number of potential destinations relative to actual samples


## Movement Probability Model

**Lead:** Train log-odds models to predict movement choice probabilities from environmental features
**Development:**
- Log-odds models enable neural network application to choice modeling with smaller datasets compared to traditional approaches (Gietzmann-Sanders, in review)
- MIMIC framework (Gietzmann-Sanders, in review) provides the implementation structure for log-odds models with systematic exploration of feature space and hyperparameters
- Models predict likelihood of moving from origin H3 cell to each potential destination given environmental features
- Training process supports both full models and incrementally complex models for feature importance analysis

**Lead:** Evaluate model architectures on validation data and select best-performing model using held-out testing set
**Development:**
- Complexity increased incrementally to maintain model interpretability and identify parsimonious feature sets
- Validation set used to evaluate each complexity level and guide feature inclusion decisions
- Complexity increased until validation performance degrades (indicating overfitting)
- All models below overfitting threshold evaluated on held-out testing set to assess generalization and prevent overfitting during model selection itself
- Best performer on testing set selected as final model for subsequent distribution inference

**Lead:** Assess feature predictivity by training models with incrementally added features
**Development:**
- Incremental addition approach quantifies how much predictive power each feature contributes beyond those already included, preventing assumption that all features contribute equally
- Features evaluated in order of their iterative contribution: first identifying the single most predictive feature, then adding the feature that most improves performance when combined with existing features, and so on
- Building from single features upward reveals which features can predict movement independently versus those that only contribute in combination with others
- Performance trajectory as features are added demonstrates diminishing returns and helps identify when additional features provide minimal benefit

**Lead:** Summarize performance metrics across selected and incremental models
**Development:**
- Performance table comparing predictive accuracy across all model architectures
- Metrics include training, validation, and testing set performance for selected model to demonstrate generalization and absence of overfitting
- Incremental model performance shows contribution of each additional feature
- Demonstrates systematic improvement and quantifies predictive capability

**Lead:** Visualize relationships between environmental features, observed movements, and model predictions
**Development:**
- Plots showing how predicted movement probabilities vary with key environmental features to validate model behavior matches ecological expectations
- Comparison of predicted versus observed movement patterns across feature gradients
- Demonstrates model captures realistic environmental responses and identifies where predictions align with or diverge from observations, building confidence in model predictions and revealing potential limitations


## Bycatch Risk Maps

**Lead:** Process and standardize Gulf of Alaska pollock fleet catch data for risk modeling
**Development:**
- Bycatch binned to Uber H3 resolution 4 and daily temporal resolution to match movement model spatiotemporal scale
- Two standardization approaches: by effort alone (standard CPUE), and by effort weighted by depth-occupancy likelihood to incorporate additional information about vertical distribution
- Depth weighting accounts for pollock trawls occurring primarily near the seafloor (De Robertis, 2006), where predicted Chinook depth occupancy affects encounter probability
- Binary catch/no-catch dataset tests whether relative risk of any bycatch can be predicted, while normalized catch quantity dataset tests whether bycatch levels can be differentiated
- Datasets split into training, validation, and testing sets

**Lead:** Summarize bycatch data coverage and characteristics across space and time
**Development:**
- Summary table showing number of fishing events, spatial coverage, and temporal coverage
- Maps showing distribution of bycatch events across Gulf of Alaska
- Demonstrates data availability for training risk model and identifies coverage gaps that may limit prediction reliability in certain regions or time periods

**Lead:** Generate movement probability matrices and derive attractor indices for bycatch locations
**Development:**
- Environmental features computed across entire Gulf of Alaska for trawl data time periods ±1 month to enable rolling window analysis
- Movement probability matrices predicted for same spatiotemporal extent
- For each time point, uniform initial distribution propagated forward through movement model to observe flow patterns without spatial bias, revealing likely areas of fish accumulation based on predicted movement alone
- Propagation performed for 1, 2, 3, and 4 weeks to assess sensitivity across within-season time horizons (week to month), with flexibility to adjust windows based on findings
- Resulting density at each cell represents "attractor index" (likelihood of fish accumulation)
- Each trawl event paired with attractor index of its location, as only relative values matter when modeling flows rather than absolute densities

**Lead:** Construct training data from paired fishing events compared on bycatch outcomes
**Development:**
- Pairs selected to be as close as possible in time and space, with proximity constrained by data availability
- Temporal and spatial separation between pairs determined by data density (closer spacing where data is dense, wider spacing in sparse regions)
- Equal representation of: both have bycatch, only one has bycatch, neither has bycatch
- Balanced representation prevents model from defaulting to always predicting risk differences and ensures ability to identify when risk levels are actually similar, supporting reliable decision-making

**Lead:** Train decision tree to predict relative bycatch risk using attractor-derived features
**Development:**
- Decision trees selected for extreme parsimony, enabling full explanation of decision-making process with attractor index features; alternative models considered if predictive performance is insufficient
- Loss function minimizes expected bycatch given model-predicted probabilities, aligning model training with management objective of directing fishing effort toward lower-risk decisions
- Features constructed exclusively from attractor indices (no direct environmental features) to test whether movement-based fish accumulation patterns correlate with observed bycatch levels and thus salmon abundance
- Validation set used to explore features, model structures, and hyperparameters

**Lead:** Evaluate bycatch risk model performance on held-out testing data
**Development:**
- Performance metrics comparing predictions to observed bycatch on testing set
- Results reported relative to 50/50 random baseline, representing the paired comparison null hypothesis where the model cannot discriminate between locations
- Confusion matrix or similar metrics showing model's ability to discriminate high versus low risk scenarios
- Demonstrates model provides meaningful risk predictions beyond chance

**Lead:** Generate interpretable spatial risk maps for specific time periods
**Development:**
- Model generates pairwise distances (difference in assigned probabilities) for all cells at interesting time points, where larger differences indicate greater divergence in expected bycatch between locations
- Agglomerative clustering groups cells with similar risk levels using the distance matrix, with flexibility to explore alternative distance-based clustering methods if needed
- Clusters assigned colors to create discrete risk zones (minimally green/yellow/red for low/medium/high risk) that are immediately interpretable for decision-making
- Visualizations demonstrate spatial structure of bycatch risk as predicted from movement patterns
