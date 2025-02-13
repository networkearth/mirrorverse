# Depth Model v2

## Week of February 14, 2025

### Confirmed Null Model

Using only `n_depth_bin` as a feature we were able to build a model that replicates the rates of the depth bins at large with no other information. 

| depth_bin | _selected rate | probability from model | 
| --- | --- | --- |
| 25 | 0.464 | 0.487 |
| 50 | 0.166 | 0.161 | 
| 75 | 0.134 | 0.125 | 
| 100 | 0.107 | 0.118 | 
| 150 | 0.130 | 0.110 | 
| 200 | 0.049 | 0.048 | 
| 250 | 0.013 | 0.011 | 
| 300 | 0.006 | 0.003 | 
| 400 | 0.002 | 0.001 | 
| 500 | 0.000 | 0.000 | 

This was with model `ab17d4ce30981b9d7630da4d7adbf7fd7cb88a9bfee2b37ed60254e097e8ffdc` in `3.1.1`. (NLP-C of 0.525) Which had the following configuration:

```python
{'batch_size': 40000,
 'dropout': 0,
 'epochs': 25,
 'layer_size': 16,
 'layers': ['D16', 'D16', 'D16'],
 'learning_rate': 0.001,
 'num_layers': 3,
 'optimizer': 'Adam',
 'optimizer_kwargs': {'learning_rate': 0.001},
 'random_seed': 2}
```

It does not seem like batch size makes much a of a difference at this point as we got equivalent scores at a batch size of 5000. 

Moreover 3000, 4000, and 5000 decisions per individual had largely the same score. The 3000 runs took 1:47, the 4000 runs took 2:35, and the 5000's took 3:07 (this was all for 25 epochs). What I'll do is run the 3000's to help me explore more quickly and then at end of day we can kick off some 5k's to see how well this model can learn with more data. 

Just for reference a true null model (log odds always 1.0) would give us a NLP-D of -1.80 whereas this model is giving us a NLP-D of -1.43, i.e. we are ~1.45x more likely to pick the correct answer. 

