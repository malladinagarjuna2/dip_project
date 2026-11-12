## Table - primary dataset (cf. paper Tables 1-3)

| display                            | classifier   |   n_features |   accuracy |   sensitivity |   specificity |   precision |     f1 |    auc |   auc_paper |   jaccard |   dice |
|:-----------------------------------|:-------------|-------------:|-----------:|--------------:|--------------:|------------:|-------:|-------:|------------:|----------:|-------:|
| Paper CNN fusion (2022)            | SVM          |          128 |     0.9586 |        0.9479 |        0.9649 |      0.9405 | 0.9442 | 0.9897 |      0.9442 |    0.8943 | 0.9442 |
| Paper CNN fusion (2022)            | XGBoost      |          128 |     0.959  |        0.952  |        0.9632 |      0.938  | 0.945  | 0.9908 |      0.945  |    0.8957 | 0.945  |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          |          128 |     0.9803 |        0.9763 |        0.9827 |      0.9706 | 0.9735 | 0.9974 |      0.9735 |    0.9483 | 0.9735 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      |          128 |     0.9802 |        0.9762 |        0.9825 |      0.9704 | 0.9733 | 0.9984 |      0.9733 |    0.9479 | 0.9733 |

## Parameter / complexity comparison

| display                            |   raw_dim |   feat_dim |   train_s |   params_M |   trainable_params_M |   gflops_per_roi |   latency_ms_per_roi |
|:-----------------------------------|----------:|-----------:|----------:|-----------:|---------------------:|-----------------:|---------------------:|
| Paper CNN fusion (2022)            |       128 |        128 |      0    |      0.005 |                0     |            0.167 |                1.101 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) |       128 |        128 |    201.12 |     49.94  |                0.492 |           33.351 |               27.122 |
