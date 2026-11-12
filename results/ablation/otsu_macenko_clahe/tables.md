## Table - primary dataset (cf. paper Tables 1-3)

| display                            | classifier   |   n_features |   accuracy |   sensitivity |   specificity |   precision |     f1 |    auc |   auc_paper |   jaccard |   dice |
|:-----------------------------------|:-------------|-------------:|-----------:|--------------:|--------------:|------------:|-------:|-------:|------------:|----------:|-------:|
| Paper CNN fusion (2022)            | SVM          |          128 |     0.9366 |        0.9479 |        0.9249 |      0.9293 | 0.9385 | 0.9821 |      0.9386 |    0.8841 | 0.9385 |
| Paper CNN fusion (2022)            | XGBoost      |          128 |     0.9365 |        0.9498 |        0.9227 |      0.9275 | 0.9385 | 0.9843 |      0.9386 |    0.8841 | 0.9385 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          |          128 |     0.9793 |        0.9784 |        0.9803 |      0.981  | 0.9797 | 0.9979 |      0.9797 |    0.9601 | 0.9797 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      |          128 |     0.9788 |        0.978  |        0.9797 |      0.9804 | 0.9792 | 0.9983 |      0.9792 |    0.9592 | 0.9792 |

## Parameter / complexity comparison

| display                            |   raw_dim |   feat_dim |   train_s |   params_M |   trainable_params_M |   gflops_per_roi |   latency_ms_per_roi |
|:-----------------------------------|----------:|-----------:|----------:|-----------:|---------------------:|-----------------:|---------------------:|
| Paper CNN fusion (2022)            |       128 |        128 |     0     |      0.005 |                0     |            0.167 |                1.035 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) |       128 |        128 |   197.752 |     49.94  |                0.492 |           33.351 |               26.885 |
