## Table - primary dataset (cf. paper Tables 1-3)

| display                            | classifier   |   n_features |   accuracy |   sensitivity |   specificity |   precision |     f1 |    auc |   auc_paper |   jaccard |   dice |
|:-----------------------------------|:-------------|-------------:|-----------:|--------------:|--------------:|------------:|-------:|-------:|------------:|----------:|-------:|
| Paper CNN fusion (2022)            | SVM          |          128 |     0.9537 |        0.9855 |        0.8303 |      0.9576 | 0.9713 | 0.9723 |      0.9715 |    0.9443 | 0.9713 |
| Paper CNN fusion (2022)            | XGBoost      |          128 |     0.9551 |        0.9841 |        0.8424 |      0.9604 | 0.9721 | 0.9752 |      0.9723 |    0.9457 | 0.9721 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          |          128 |     0.9688 |        0.9886 |        0.892  |      0.9727 | 0.9806 | 0.9939 |      0.9806 |    0.9619 | 0.9806 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      |          128 |     0.9688 |        0.9885 |        0.8922 |      0.9727 | 0.9806 | 0.9945 |      0.9806 |    0.9618 | 0.9806 |

## Parameter / complexity comparison

| display                            |   raw_dim |   feat_dim |   train_s |   params_M |   trainable_params_M |   gflops_per_roi |   latency_ms_per_roi |
|:-----------------------------------|----------:|-----------:|----------:|-----------:|---------------------:|-----------------:|---------------------:|
| Paper CNN fusion (2022)            |       128 |        128 |      0    |      0.005 |                0     |            0.167 |                1.186 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) |       128 |        128 |    190.46 |     49.94  |                0.492 |           33.351 |               27.628 |
