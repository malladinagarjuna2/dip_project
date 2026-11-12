## Table - primary dataset (cf. paper Tables 1-3)

| display                            | classifier   |   n_features |   accuracy |   sensitivity |   specificity |   precision |     f1 |    auc |   auc_paper |   jaccard |   dice |
|:-----------------------------------|:-------------|-------------:|-----------:|--------------:|--------------:|------------:|-------:|-------:|------------:|----------:|-------:|
| Paper CNN fusion (2022)            | SVM          |          128 |     0.9391 |        0.9402 |        0.9384 |      0.9054 | 0.9225 | 0.9827 |      0.9228 |    0.8561 | 0.9225 |
| Paper CNN fusion (2022)            | XGBoost      |          128 |     0.9378 |        0.942  |        0.9352 |      0.9012 | 0.9211 | 0.9853 |      0.9216 |    0.8538 | 0.9211 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          |          128 |     0.9798 |        0.9742 |        0.9833 |      0.9733 | 0.9738 | 0.9972 |      0.9738 |    0.9489 | 0.9738 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      |          128 |     0.9793 |        0.9732 |        0.9832 |      0.9732 | 0.9732 | 0.9983 |      0.9732 |    0.9478 | 0.9732 |

## Parameter / complexity comparison

| display                            |   raw_dim |   feat_dim |   train_s |   params_M |   trainable_params_M |   gflops_per_roi |   latency_ms_per_roi |
|:-----------------------------------|----------:|-----------:|----------:|-----------:|---------------------:|-----------------:|---------------------:|
| Paper CNN fusion (2022)            |       128 |        128 |     0     |      0.005 |                0     |            0.167 |                1.134 |
| Hybrid ConvNeXt+DINOv2+CBAM (ours) |       128 |        128 |   205.357 |     49.94  |                0.492 |           33.351 |               27.711 |
