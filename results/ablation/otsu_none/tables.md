## Table - primary dataset (cf. paper Tables 1-3)

| display                            | classifier   |   n_features |   accuracy |   sensitivity |   specificity |   precision |     f1 |    auc |   auc_paper |   jaccard |   dice |
|:-----------------------------------|:-------------|-------------:|-----------:|--------------:|--------------:|------------:|-------:|-------:|------------:|----------:|-------:|
| Paper CNN fusion (2022)            | SVM          |          128 |     0.9556 |        0.9726 |        0.9097 |      0.9667 | 0.9696 | 0.9881 |      0.9696 |    0.9411 | 0.9696 |
| Paper CNN fusion (2022)            | XGBoost      |          128 |     0.9555 |        0.9713 |        0.9129 |      0.9678 | 0.9696 | 0.9897 |      0.9696 |    0.941  | 0.9696 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          |          128 |     0.9779 |        0.9856 |        0.9573 |      0.9842 | 0.9849 | 0.9969 |      0.9849 |    0.9702 | 0.9849 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      |          128 |     0.9782 |        0.9855 |        0.9584 |      0.9846 | 0.985  | 0.9976 |      0.985  |    0.9705 | 0.985  |

## Parameter / complexity comparison

| display                            |   raw_dim |   feat_dim |   train_s |   params_M |   trainable_params_M |   gflops_per_roi |   latency_ms_per_roi |
|:-----------------------------------|----------:|-----------:|----------:|-----------:|---------------------:|-----------------:|---------------------:|
| Paper CNN fusion (2022)            |       128 |        128 |     0     |      0.005 |                0     |            0.167 |                0.972 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) |       128 |        128 |   184.482 |     49.94  |                0.492 |           33.351 |               26.632 |
