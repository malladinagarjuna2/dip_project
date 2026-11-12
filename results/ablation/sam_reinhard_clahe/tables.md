## Table - primary dataset (cf. paper Tables 1-3)

| display                            | classifier   |   n_features |   accuracy |   sensitivity |   specificity |   precision |     f1 |    auc |   auc_paper |   jaccard |   dice |
|:-----------------------------------|:-------------|-------------:|-----------:|--------------:|--------------:|------------:|-------:|-------:|------------:|----------:|-------:|
| Paper CNN fusion (2022)            | SVM          |          128 |     0.9502 |        0.9477 |        0.9518 |      0.9285 | 0.938  | 0.9895 |      0.9381 |    0.8833 | 0.938  |
| Paper CNN fusion (2022)            | XGBoost      |          128 |     0.9544 |        0.9493 |        0.9578 |      0.9368 | 0.943  | 0.9908 |      0.9431 |    0.8922 | 0.943  |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          |          128 |     0.9799 |        0.979  |        0.9805 |      0.9708 | 0.9749 | 0.9974 |      0.9749 |    0.951  | 0.9749 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      |          128 |     0.9801 |        0.9784 |        0.9813 |      0.9718 | 0.9751 | 0.9978 |      0.9751 |    0.9514 | 0.9751 |

## Parameter / complexity comparison

| display                            |   raw_dim |   feat_dim |   train_s |   params_M |   trainable_params_M |   gflops_per_roi |   latency_ms_per_roi |
|:-----------------------------------|----------:|-----------:|----------:|-----------:|---------------------:|-----------------:|---------------------:|
| Paper CNN fusion (2022)            |       128 |        128 |     0     |      0.005 |                0     |            0.167 |                0.888 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) |       128 |        128 |   193.017 |     49.94  |                0.492 |           33.351 |               26.965 |
