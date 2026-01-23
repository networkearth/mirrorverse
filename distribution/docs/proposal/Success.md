# Success Criteria

## Migration Classification Model

**Goal**: Ensure the migration classifier accurately distinguishes migration from non-migration periods while minimizing contamination of the non-migration training dataset.

**Criteria**: The classifier achieves a false negative rate below 5% while maintaining a true negative rate above 75% on the validation set.

## Movement Probability Model

**Goal**: Develop a parsimonious and generalizable movement probability model that predicts Chinook salmon movement choices based on environmental features.

**Criteria**: The selected model demonstrates consistent performance across training, validation, and testing sets according to the negative log likelihood of the data given the predicted probabilities from the model.

## Bycatch Risk Maps

**Goal**: Predict relative bycatch risk from movement-based fish accumulation patterns to support fishing effort decisions that minimize Chinook salmon bycatch.

**Criteria**: The decision tree model discriminates between high and low bycatch risk scenarios on held-out testing data at a level above a 50/50 random baseline.