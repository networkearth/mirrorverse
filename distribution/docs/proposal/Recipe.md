## Validated Tag Movement Data
**Lead:** Use pop-up satellite tag data from Chinook salmon in the Gulf of Alaska
**Development:**
- We will use existing tag deployment data (Seitz, 2024).
- Fish were captured by hook and line.
- Deployments spanned multiple locations: Dutch Harbor, AK (n=20), Chignik, AK (n=16), Craig, AK (n=8), Homer, AK (n=20), Kodiak, AK (n=13), Yakutat, AK (n=16), Sitka, AK (n=15), and Central Bering Sea (n=3)
- Only healthy individuals meeting minimum size criteria (62-100cm fork length) were selected for tagging
- Pop-up satellite archival tags attached following established protocols (Seitz, 2024)
- Tagged fish released at capture location

**Lead:** Collect time-series depth, temperature, and position data from surfaced tags
**Development:**
- The tags collected temperature, ambient light intensity, and depth information at sub-daily intervals during deployment
- These measurements enabled light-based geolocation (using sunrise and sunset events along with a GMT clock to estimate position) with temperature-based corrections that refine estimates by matching recorded temperatures to known sea surface temperature fields
- Tags were programmed to release from fish, surface, and transmit collected data over satellite
- Data were transmitted in time-series format with measurements at 2.5-15 minute intervals before uploading

**Lead:** Convert raw tag transmissions to spatially-gridded position estimates at H3 resolution 4
**Development:**
- Likely path estimates were obtained using a proprietary geolocation algorithm from Wildlife Computers (Wildlife Computers, 2025)
- An estimate of location for each day was then derived as the central tendency of those paths
- Position estimates will be converted to Uber H3 hexagonal grid cells at resolution 4 (~26km edge length), a hierarchical spatial indexing system that provides uniform cell sizes across the globe
- Aggregation to H3 resolution 4 accommodates geolocation uncertainty while maintaining meaningful spatial resolution, creating spatially-standardized movement tracks

**Lead:** Filter tag records to retain only post-acclimation movement data
**Development:**
- Fish behavior immediately following tagging may reflect handling stress rather than natural movement
- All movement data from first seven days post-deployment excluded from analysis to allow fish to return to normal behavioral patterns

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

## Homing Migration Classification and Filtering

**Lead:** Salmon homing migrations (return journeys to natal streams) must be separated from routine at-sea movements to avoid biasing our analysis of distribution preferences (Beamish, 2018).
**Development:**
- Therefore we will need to build a model that allows us to distinguish, across the dataset, between migrating fish and non-migrating fish. 

**Lead:** First we'll need a dataset to train the model.
**Development:**
- Michael Courtney and Andrew Seitz will manually label unambiguous migration and non-migration periods, as no standard methodology exists for this classification
- Clear positive and negative examples establish the training foundation, while the classifier will handle ambiguous intermediate cases

**Lead:** Visualize example trajectories with classified migration and non-migration segments
**Development:**
- Map examples showing trajectories with migration segments (red) and non-migration segments (blue) overlaid
- Examples selected to illustrate range of movement patterns observed in classified data
- Demonstrates classifier successfully distinguishes migration patterns in geospatial context

**Lead:** Next we'll derive temporal movement features using bidirectional time windows
**Development:**
- Temporal windows examine both forward and backward in time from point of interest
- Window sizes constrained to one week or less to help retain data from fish with shorter tagging periods
- Features extracted from movement data only (no environmental features), as migration should be identifiable from the nature of the movement alone
- Feature types include momentum (sustained directional movement), behavioral consistency (erratic versus steady patterns), and temporal continuity (persistence across window)

**Lead:** Then we'll train a supervised classifier to predict migration periods from movement features
**Development:**
- Model predicts migration probability given features from bidirectional temporal window
- We will select a classification threshold (operating point) where false negatives (migration points incorrectly labeled as non-migration) are expected to be below 5% on the validation dataset
- We will then evaluate the true negative rate (non-migration points correctly labeled) at that threshold — higher is better, as it means we retain more non-migration samples without accidentally including migration data
- Class balancing applied during training to ensure equal representation of migration and non-migration periods
- Candidate models include random forests, gradient-boosted trees (XGBoost), and neural networks, with flexibility to explore other architectures if needed

**Lead:** Show classifier performance
**Development:**
- Confusion matrix (table showing correct vs. incorrect classifications) displaying true/false positives and negatives on validation set
- ROC (Receiver Operating Characteristic) curves, which plot true positive rate against false positive rate across all possible thresholds, for both training and testing sets to assess generalization and identify overfitting

**Lead:** Apply the trained classifier to isolate non-migratory movements from the full dataset
**Development:**
- All unlabeled time points classified using selected operating point
- Points classified as migration removed from dataset
- Resulting dataset contains only non-migration movements for movement model training
- Goal is dataset where migration movements have been filtered out with high confidence

**Lead:** Summarize results of filtering.
**Development:**
- Summary table reporting data filtered: total time points, points classified as migration, points classified as non-migration
- Individual-level statistics showing distribution of migration periods per fish


## Feature-Enriched Training Data

**Lead:** Develop and categorize comprehensive list of candidate environmental features that could be used in movement modeling
**Development:**
- Exhaustive feature list compiled from published literature and expert panel input (Andrew Seitz, Michael Courtney, Curry Cunningham) to ensure consideration of features beyond those typically used
- Features categorized as: stationary (unchanging features like bathymetry), temporal (features that vary with time like temperature), or stateful (features that accumulate or have memory like productivity/chlorophyll)
- We will pull such features from ocean models and remote sensing platforms such as Copernicus Marine Service and Google Earth
- All features standardized to Uber H3 resolution 4 and daily temporal resolution for joining with movement data

**Lead:** Cluster features by information content using vector embedding analysis
**Development:**
- Highly correlated features can cause overfitting, so dimensionality reduction techniques are used to identify redundant information
- Vector embeddings use neural networks to learn non-linear relationships between features, unlike PCA (Principal Component Analysis) which only captures linear relationships
- We will use embeddings to reduce our exhaustive feature set to a smaller set, allowing us to explore the feature space more efficiently
- However, embedded features lose their physical interpretability
- To restore interpretability, we will identify which original features best predict the embedded features
- These representative features will become our modeling candidates 

**Lead:** Normalize and scale representative features for neural network compatibility
**Development:**
- Neural networks learn by adjusting weights based on feature values; features with larger numeric ranges can dominate the learning process if not normalized
- Features rescaled to [0,1] if non-negative (e.g., depth, distance) or [-1,1] if containing negative values (e.g., temperature anomalies), ensuring all features contribute equally during model training

**Lead:** Resample movement data to ensure balance across space, time, and individual fish
**Development:**
- Study area divided into four spatial regions: Aleutian Island Chain, Kodiak to Yakutat, Southeast Alaska (to British Columbia border), and British Columbia and south
- Spatial regions correspond with typical fisheries management boundaries in the Gulf of Alaska
- Each spatial region further divided into seasonal blocks (Winter, Spring, Summer, Fall) as Chinook behavior varies seasonally (Gietzmann-Sanders, in review; Seitz, 2024) and finer temporal resolution would spread data too thinly
- Samples collected into each space-time block with equal representation across all blocks
- Within each block, resample with replacement the data from each individual to ensure all individuals in each area/time are equally represented. 

**Lead:** Create examples from our observations and join to our features to create our final dataset
**Development:**
- First we'll calculate the 90th percentile range of movement per day per fish.
- Then we'll create examples from each movement that include: origin location, actual destination, and all potential destinations (h3 resolution 4 cells) within the 90th percentile daily range to capture realistic movement options
- Actual destination will then be labeled as "selected"
- We will then join these positions and times to their corresponding features as described above


## Movement Probability Model

**Lead:** We will pose our movement model as a classification problem, predicting where a fish will go based on the environmental features at potential destinations (including staying in place).
**Development:**
- Classification models output the probability of selecting each destination, which we can use to build a transition matrix (a table showing the probability of moving from any location to any other location)
- This transition matrix tells us how fish density at any point in our grid will redistribute in the next timestep — exactly what we need for tracking population distribution
- This approach is more computationally efficient than traditional particle tracking, which simulates many individual fish trajectories
- Instead of tracing thousands of individual movements, we simply track how overall density shifts over time 

**Lead:** We will use a log-odds modeling framework to predict these movement choices
**Development:**
- With many potential destinations, each having its own environmental features, the total number of model inputs grows very large
- High-dimensional feature spaces require exponentially more data to fit effectively — a problem known as the "curse of dimensionality" (Verleysen, 2005)
- Log-odds models address this by evaluating each destination's features independently rather than jointly, reducing dimensionality from (features × destinations) to just (features)
- This dramatically reduces data requirements and overfitting risk (Gietzmann-Sanders, in review)
- We will use the MIMIC framework (Gietzmann-Sanders, in review), which implements log-odds modeling for movement prediction

**Lead:** As the first step we will reorganize our input data in preparation for training.
**Development:**
- Data will be split into training, testing, and validation sets
- Each individual fish is assigned to only one set (no fish split across sets), preventing data leakage where the model learns patterns from the same fish it will be tested on
- We will employ contrast resampling (Gietzmann-Sanders, in review), a technique required for log-odds modeling that structures data as pairwise comparisons
- If after this we observe high biases towards specific regions/times in any of the datasets we will consider skipping the unbiasing earlier and apply it at this point instead. 

**Lead:** We will then train and hyperparameter tune models across our feature set.
**Development:**
- Hyperparameter tuning (adjusting model settings like learning rate and network size) will be driven by performance on the validation set
- The test set is held out until final evaluation to provide an unbiased estimate of model performance
- Performance will be measured by negative log likelihood (how well the model's predicted probabilities match observed movements)
- The model that performs best on the validation set will be selected

**Lead:** Assess feature predictivity by training models with incrementally added features
**Development:**
- We start with a null model (predicting movement without any features) and add features one at a time
- At each step, we fit models with each remaining feature and select the one that improves validation performance most
- This continues until we arrive at our selected model
- The incremental approach quantifies how much predictive power each feature contributes beyond features already included
- Each incremental model will be evaluated on the held-out test set for final performance assessment
- The most predictive model will be selected for subsequent analyses 

**Lead:** Summarize performance metrics across selected and incremental models
**Development:**
- Performance table comparing predictive accuracy across the incremental models
- Metrics include training, validation, and testing set performance for the selected model to demonstrate generalization (ability to predict new data) and absence of overfitting (memorizing training data rather than learning patterns)

**Lead:** Visualize relationships between environmental features, observed movements, and model predictions
**Development:**
- Plots showing how predicted movement probabilities vary with key environmental features to validate that model behavior matches ecological expectations
- Comparison of predicted versus observed movement patterns across feature gradients
- These visualizations will allow us to better understand how what the model is learning is related to what is understood about the ecology and physiology of Chinook salmon.


## Bycatch Risk Maps

**Lead:** Process and standardize Gulf of Alaska pollock fleet catch data for risk modeling
**Development:**
- Bycatch binned to Uber H3 resolution 4 and daily temporal resolution to match movement model spatiotemporal scale
- Two standardization approaches: by effort alone (standard CPUE, or catch-per-unit-effort), and by effort weighted by depth-occupancy likelihood (how often fish are predicted to be at trawl depths) (Gietzmann-Sanders, in review)
- Depth weighting accounts for pollock trawls occurring primarily near the seafloor (De Robertis, 2006), where predicted Chinook depth occupancy affects encounter probability
- From this we create two datasets: a binary catch/no-catch dataset to test whether any bycatch can be predicted, and a normalized catch quantity dataset to test whether bycatch levels can be differentiated

**Lead:** Summarize bycatch data coverage and characteristics across space and time
**Development:**
- Summary table showing number of fishing events, spatial coverage, and temporal coverage
- Maps showing distribution of bycatch events across Gulf of Alaska
- Demonstrates data availability for training risk model and identifies coverage gaps that may limit prediction reliability in certain regions or time periods

**Lead:** Generate movement probability matrices and derive attractor indices for bycatch locations
**Development:**
- We use density tracking to identify basins of attraction (areas where fish are predicted to accumulate over time)
- Model features computed across entire Gulf of Alaska for trawl data time periods ±1 month to enable rolling window analysis
- Use these model features to predict transition matrices per day
- For each time point in the bycatch data, move back 1 week and start with a uniform initial distribution (equal probability across all cells)
- Then propagate that forward using the transition matrices, revealing likely areas of fish accumulation
- Resulting density per H3 cell represents an "attractor index" (higher values indicate areas more likely to attract fish)
- We repeat this for 2 weeks prior, 3 weeks, and 4 weeks to assess performance across different time horizons (week to month), with flexibility to adjust windows based on findings
- In the end we will have paired each trawl event with attractor indices derived from these 1, 2, 3, and 4 week time windows

**Lead:** Construct training data from paired fishing events compared on bycatch outcomes
**Development:**
- Given attractor indices only have meaning relative to other nearby H3 cells, we pair fishing events that are reasonably close in space and time (constrained by data availability)
- We want equal representation of three outcomes: both events have bycatch, only one has bycatch, and neither has bycatch
- This balance prevents the model from defaulting to always predicting risk differences, ensuring it can identify when risk levels are actually similar
- We then divide these pairs into training, validation, and testing sets

**Lead:** Train decision tree to predict relative bycatch risk using attractor-derived features
**Development:**
- We set up the problem as predicting which location in each pair is the better choice (results in less bycatch) given only the pair's attractor indices
- We use decision trees for their simplicity and interpretability, enabling full explanation of the decision-making process; alternative models considered if predictive performance is insufficient
- We use a loss function (used to guide the training of the model) based on expected bycatch given the probabilities the model assigns to each decision
- Minimizing this loss creates a decision tree that minimizes expected bycatch over the training samples
- Validation set used to explore features, model structures, and hyperparameters

**Lead:** Evaluate bycatch risk model performance on held-out testing data
**Development:**
- Performance metrics comparing predictions to observed bycatch on testing set
- Results reported relative to 50/50 random baseline, representing the null hypothesis (what we'd expect if the model had no predictive ability)
- Would demonstrated the model provides meaningful risk predictions beyond chance

**Lead:** Generate interpretable spatial risk maps for specific time periods
**Development:**
- The model generates pairwise distances (difference in assigned probabilities) between all cells, which we use to group similar areas
- Agglomerative clustering (a method that progressively merges similar items) groups cells with similar risk levels using these distances, with flexibility to explore alternative clustering methods if needed
- Clusters assigned colors to create discrete risk zones (minimally green/yellow/red for low/medium/high risk) that are immediately interpretable for decision-making
- Visualizations demonstrate spatial structure of bycatch risk as predicted from movement patterns
