## Validated Tag Movement Data
**Lead:** Use pop-up satellite tag data from Chinook salmon in the Gulf of Alaska
**Development:**
- We will use existing tag deployment data.
- Fish were captured by hook and line.
- Deployments spanned multiple locations: Dutch Harbor, AK (n=20), Chignik, AK (n=16), Craig, AK (n=8), Homer, AK (n=20), Kodiak, AK (n=13), Yakutat, AK (n=16), Sitka, AK (n=15), and Central Bering Sea (n=3)
- Only healthy individuals meeting minimum size criteria (62-100cm fork length) were selected for tagging
- Pop-up satellite archival tags attached following established protocols
- Tagged fish released at capture location

**Lead:** Collect time-series depth, temperature, and position data from surfaced tags
**Development:**
- The tags used collected temperature, ambient light intensity, and depth information at specified (sub/day) intervals during deployment - these measurements enabled light-based geolocation with temperature-based corrections using surface temperature data
- Tags were programmed to release from fish, surface, and transmit collected data over satellite
- Data were transmitted in time-series format with measurements at 2.5-15 minute intervals before uploading

**Lead:** Convert raw tag transmissions to spatially-gridded position estimates at H3 resolution 4
**Development:**
- To obtain location estimates, the data from the tag was passed through a proprietary hidden Markov model from Wildlife Computers that used the tag’s data to estimate the likely paths taken by the fish during tag deployment (Wildlife Computers, 2025). 
- An estimate of location for each day was then derived as the central tendency of those paths on a particular day. 
- Position estimates will be converted to Uber H3 hierarchical hexagonal grid at resolution 4 (~26km edge length hexagons)
- Aggregation to H3 resolution 4 accommodates geolocation uncertainty while maintaining meaningful spatial resolution creating spatially-standardized movement tracks

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

## Migration Classification and Filtering

**Lead:** We do not want to confound the migrations that salmon undergo (Beamish, 2018) with the movement patterns that will enable us to identify distribution preferences.
**Development:**
- Therefore we will need to build a model that allows us to distinguish, across the dataset, between migrating fish and non-migrating fish. 

**Lead:** First we'll need a dataset to train the model.
**Development:**
- Michael Courtney and Andrew Seitz will identify unambiguous migration periods and unambiguous non-migration periods through manual expert labeling, as no standard methodology exists for this classification
- Clear positive and negative examples establish training foundation while classifier will be able to handle ambiguous intermediate cases

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
- Model performance evaluated by first identifing an operating point at which false negatives (falsely labelled migration points) are expected to be below 5% on the validation dataset and then evaluating the true negative rate (correctly labelled non migration points). 
- The higher the true negative rate the better as it'll mean we'll be able to keep a large number of non-migration samples without accidently polluting our dataset with migration samples.
- Class balancing applied to ensure equal representation of migration and non-migration periods during training
- Candidate models will include random forests, XGBoost, and neural networks (other architectures considered if issues arise)

**Lead:** Show classifier performance
**Development:**
- Confusion matrix showing true/false positives and negatives on validation set
- ROC curves for both training and testing sets to assess generalization and identify overfitting

**Lead:** Apply the trained classifier to isolate non-migratory movements from the full dataset
**Development:**
- All unlabeled time points classified using selected operating point
- Points classified as migration removed from dataset
- Resulting dataset contains only non-migration movements for movement model training
- Goal is dataset where migration movements have been filtered out with high confidence

**Lead:** Summarize results of filtering.
- Summary table reporting data filtered: total time points, points classified as migration, points classified as non-migration
- Individual-level statistics showing distribution of migration periods per fish


## Feature-Enriched Training Data

**Lead:** Develop and categorize comprehensive list of candidate environmental features that could be used in movement modeling
**Development:**
- Exhaustive feature list compiled from published literature and expert panel input (Andrew Seitz, Michael Courtney, Curry Cunningham) to ensure consideration of features beyond those typically used
- Features categorized as: stationary (bathymetry), temporal (temperature), or stateful (productivity/chlorophyll)
- We will pull such features from ocean models such as Copernicus Marine Service
- All features standardized to Uber H3 resolution 4 and daily temporal resolution for joining with movement data

**Lead:** Cluster features by information content using vector embedding analysis
**Development:**
- Dimensionality reduction techniques are normally used to ensure highly correlated features are not used in the same model.
- This helps prevent overfitting.
- Vector embeddings are a dimensionality reduction technique that takes advantage of neural networks to be able to learn non-linear relationships between features as opposed to techniques like PCA that require linear relationships.
- We will use this technique to reduce our exhaustive feature set into a set of lower cardinality thus allowing us to explore the feature space more quickly and robustly. 
- This will result however in a set of derivative features that may have little to know physical meaning.
- To restore parsimony we will then look for a subset of our original exhaustive feature set that are predictive themselves of the embedded features. 
- These representative features will then become our candidates for modelings. 

**Lead:** Normalize and scale representative features for neural network compatibility
**Development:**
- We will be using neural networks in the subsequent modeling and thus require normalizing the ranges of our features as to not bias gradient descent toward specific features.
- Features rescaled to [0,1] if non-negative (representing scales or magnitudes) or [-1,1] if containing negative values, keeping all features in similar ranges to ensure similar gradient magnitudes across features during optimization

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

**Lead:** We will first pose our movement model as a classification problem where we will try to predict where a fish will go given the features at the potential destinations (which include staying put).
- Classification models such as these give us the likelihoods of picking any specific class - in this case the likelihoods of the various destinations.
- In turn this will allow us to use the model to build a transition matrix that will tell us how density at any point in our grid will move in the next timestep. 
- This is exactly what we need for our density tracking. 

**Lead:** We will use a log-odds modeling framework to predict these movement choices
**Development:**
- We will have a large number of potential movement choices and given each of these will be associated with its own features we will have a very large number of dimensions in our feature space.
- This is an issue because the volume of data required to fit a model effectively can grow exponentially with the dimensionality of the feature space (Verleysen, 2005). 
- By using a log-odds modeling framework we can effectively reduce the feature space dimensionality to the number of features per option as opposed to the number of features across all options (Gietzmann-Sanders, in review)
- This greatly increases our odds of being able to fit models without overfitting
- We will use the MIMIC framework (Gietzmann-Sanders, in review) as it provides an implementation for log-odds models

**Lead:** As the first step we will reorganize our input data in preparation for training.
- We will split the data into training, testing, and validation sets. 
- Individuals will be assigned to only one of the sets (no fish split across sets) to prevents data leakage from autocorrelated movements within individuals across testing and training sets
- We will then employ contrast resampling (Gietzmann-Sanders, in review) as it is a requirement for log-odds modeling.
- After after this we observe high biases towards specific regions/times in any of the datasets we will consider skipping the unbiasing earlier and apply it at this point instead. 

**Lead:** We will then train and hyperparameter tune models across our feature set.
**Development:**
- Hyperparameter tuning will be driven by performance over the validation set in order to allow us to hold-out the training set until the very end. 
- Performance will be measured by the negative log likelihood of the data given the predictions from the model. 
- The model that performs the best over the validation set will be selected.

**Lead:** Assess feature predictivity by training models with incrementally added features
**Development:**
- We will then start with a null model (no features), add the most predictive feature (by fitting with only that feature and evaluating over the validation dataset), and then continue to add features until we arrive at our selected model. 
- Incremental addition approach quantifies how much predictive power each feature contributes beyond those already 
- Building from single features upward reveals which features can predict movement independently versus those that only contribute in combination with others
- Each of these will then be evaluated across the hold out test set as well for final evaluation of the models. 
- The most predictive of these will then be selected for the subsequent analyses. 

**Lead:** Summarize performance metrics across selected and incremental models
**Development:**
- Performance table comparing predictive accuracy across the incremental models.
- Metrics include training, validation, and testing set performance for selected model to demonstrate generalization and absence of overfitting

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
- From this we will create two datasets - a binary catch/no-catch dataset to test whether relative risk of any bycatch can be predicted and then a normalized catch quantity dataset to test whether bycatch levels can be differentiated

**Lead:** Summarize bycatch data coverage and characteristics across space and time
**Development:**
- Summary table showing number of fishing events, spatial coverage, and temporal coverage
- Maps showing distribution of bycatch events across Gulf of Alaska
- Demonstrates data availability for training risk model and identifies coverage gaps that may limit prediction reliability in certain regions or time periods

**Lead:** Generate movement probability matrices and derive attractor indices for bycatch locations
**Development:**
- This is where we do our density tracking in order to understand where our basins of attraction are.
- Model features computed across entire Gulf of Alaska for trawl data time periods ±1 month to enable rolling window analysis
- Use these model features to predict transition matrices per day 
- For each time point in the bycatch data, move back 1 week and start with a uniform initial distribution. 
- Then propagate that forward using the transition matrices, revealing likely areas of fish accumulation
- Resulting density per h3 cell cell represents an "attractor index" (higher values being more likely to be areas that attract fish)
- We will repeat this for 2 weeks prior, 3 weeks, and 4 weeks to assess performance across different time horizons (week to month), with flexibility to adjust windows based on findings
- In the end we will have paired each trawl event with an attractor index derived from a 1, 2, 3, and 4 week time window.

**Lead:** Construct training data from paired fishing events compared on bycatch outcomes
**Development:**
- Given attractor indices only have meaning relative to other nearby h3 cells we'll need to pair fishing events that are reasonably close in space and time (constrained by data availability)
- We'll want equal representation of: both have bycatch, only one has bycatch, neither has bycatch in order to prevent the model from defaulting to always predicting risk differences and ensuring its ability to identify when risk levels are actually similar, supporting reliable decision-making
- We'll then divide these pairs into training, validation, and testing sets

**Lead:** Train decision tree to predict relative bycatch risk using attractor-derived features
**Development:**
- We'll setup the problem as wanting to predict which of each given pair is the better choice (results in less bycatch) given only information about the pair's attractor indices. 
- We'll use decision trees for extreme parsimony, enabling full explanation of decision-making process with attractor index features; alternative models considered if predictive performance is insufficient
- We'll use a loss function that is the expected bycatch given the probabilities the model assigns to each decision. Minimizing this loss function will create a decision tree that minimizes expected bycatch over the training samples. 
- Validation set used to explore features, model structures, and hyperparameters

**Lead:** Evaluate bycatch risk model performance on held-out testing data
**Development:**
- Performance metrics comparing predictions to observed bycatch on testing set
- Results reported relative to 50/50 random baseline, representing the paired comparison null hypothesis where the model cannot discriminate between locations
- Demonstrates model provides meaningful risk predictions beyond chance

**Lead:** Generate interpretable spatial risk maps for specific time periods
**Development:**
- Given the model effectively generates pairwise distances (difference in assigned probabilities) for all cells we can use these distances to do clustering
- Agglomerative clustering groups cells with similar risk levels using the distance matrix, with flexibility to explore alternative distance-based clustering methods if needed
- Clusters assigned colors to create discrete risk zones (minimally green/yellow/red for low/medium/high risk) that are immediately interpretable for decision-making
- Visualizations demonstrate spatial structure of bycatch risk as predicted from movement patterns
