| localizer     | preprocessing   | display                            | classifier   | dataset   |   accuracy |     f1 |   jaccard |    auc |   roi_wbc_coverage |
|:--------------|:----------------|:-----------------------------------|:-------------|:----------|-----------:|-------:|----------:|-------:|-------------------:|
| (run default) | (run default)   | Hybrid CNN-only                    | FCL          | primary   |     0.9787 | 0.9725 |    0.9464 | 0.9978 |           nan      |
| (run default) | (run default)   | Hybrid CNN-only                    | RF           | primary   |     0.9792 | 0.973  |    0.9474 | 0.9965 |           nan      |
| (run default) | (run default)   | Hybrid CNN-only                    | SVM          | primary   |     0.9791 | 0.9729 |    0.9473 | 0.9964 |           nan      |
| (run default) | (run default)   | Hybrid CNN-only                    | XGBoost      | primary   |     0.9789 | 0.9726 |    0.9467 | 0.9976 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | FCL          | primary   |     0.9798 | 0.9738 |    0.9488 | 0.9985 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | RF           | primary   |     0.9801 | 0.9742 |    0.9496 | 0.9981 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9806 | 0.9748 |    0.9508 | 0.9981 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9803 | 0.9743 |    0.95   | 0.9985 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+Swin+CBAM (ours)   | FCL          | primary   |     0.98   | 0.9741 |    0.9496 | 0.9983 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+Swin+CBAM (ours)   | RF           | primary   |     0.9801 | 0.9742 |    0.9496 | 0.9974 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+Swin+CBAM (ours)   | SVM          | primary   |     0.9803 | 0.9745 |    0.9503 | 0.9962 |           nan      |
| (run default) | (run default)   | Hybrid ConvNeXt+Swin+CBAM (ours)   | XGBoost      | primary   |     0.9802 | 0.9743 |    0.9498 | 0.9983 |           nan      |
| (run default) | (run default)   | Hybrid ViT-only                    | FCL          | primary   |     0.9819 | 0.9765 |    0.954  | 0.9988 |           nan      |
| (run default) | (run default)   | Hybrid ViT-only                    | RF           | primary   |     0.9817 | 0.9763 |    0.9536 | 0.9985 |           nan      |
| (run default) | (run default)   | Hybrid ViT-only                    | SVM          | primary   |     0.982  | 0.9766 |    0.9543 | 0.9984 |           nan      |
| (run default) | (run default)   | Hybrid ViT-only                    | XGBoost      | primary   |     0.9823 | 0.977  |    0.9551 | 0.9988 |           nan      |
| (run default) | (run default)   | Hybrid w/o attention               | FCL          | primary   |     0.9784 | 0.972  |    0.9455 | 0.9982 |           nan      |
| (run default) | (run default)   | Hybrid w/o attention               | RF           | primary   |     0.9788 | 0.9725 |    0.9464 | 0.9975 |           nan      |
| (run default) | (run default)   | Hybrid w/o attention               | SVM          | primary   |     0.9789 | 0.9727 |    0.9468 | 0.9973 |           nan      |
| (run default) | (run default)   | Hybrid w/o attention               | XGBoost      | primary   |     0.9784 | 0.972  |    0.9456 | 0.9981 |           nan      |
| (run default) | (run default)   | Hybrid w/o paper branches          | FCL          | primary   |     0.9799 | 0.9739 |    0.9492 | 0.9984 |           nan      |
| (run default) | (run default)   | Hybrid w/o paper branches          | RF           | primary   |     0.98   | 0.9741 |    0.9495 | 0.9977 |           nan      |
| (run default) | (run default)   | Hybrid w/o paper branches          | SVM          | primary   |     0.9804 | 0.9746 |    0.9504 | 0.9969 |           nan      |
| (run default) | (run default)   | Hybrid w/o paper branches          | XGBoost      | primary   |     0.9803 | 0.9744 |    0.9501 | 0.9984 |           nan      |
| (run default) | (run default)   | Paper CNN fusion (2022)            | FCL          | primary   |     0.9385 | 0.9224 |    0.8559 | 0.9841 |           nan      |
| (run default) | (run default)   | Paper CNN fusion (2022)            | RF           | primary   |     0.9363 | 0.9194 |    0.8508 | 0.9844 |           nan      |
| (run default) | (run default)   | Paper CNN fusion (2022)            | SVM          | primary   |     0.9391 | 0.9225 |    0.8561 | 0.9827 |           nan      |
| (run default) | (run default)   | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.9378 | 0.9211 |    0.8538 | 0.9853 |           nan      |
| otsu          | macenko+CLAHE   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9793 | 0.9797 |    0.9601 | 0.9979 |             0.9176 |
| otsu          | macenko+CLAHE   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9788 | 0.9792 |    0.9592 | 0.9983 |             0.9176 |
| otsu          | macenko+CLAHE   | Paper CNN fusion (2022)            | SVM          | primary   |     0.9366 | 0.9385 |    0.8841 | 0.9821 |             0.9176 |
| otsu          | macenko+CLAHE   | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.9365 | 0.9385 |    0.8841 | 0.9843 |             0.9176 |
| otsu          | none            | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9779 | 0.9849 |    0.9702 | 0.9969 |             0.6551 |
| otsu          | none            | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9782 | 0.985  |    0.9705 | 0.9976 |             0.6551 |
| otsu          | none            | Paper CNN fusion (2022)            | SVM          | primary   |     0.9556 | 0.9696 |    0.9411 | 0.9881 |             0.6551 |
| otsu          | none            | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.9555 | 0.9696 |    0.941  | 0.9897 |             0.6551 |
| otsu          | reinhard+CLAHE  | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9688 | 0.9806 |    0.9619 | 0.9939 |             0.4685 |
| otsu          | reinhard+CLAHE  | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9688 | 0.9806 |    0.9618 | 0.9945 |             0.4685 |
| otsu          | reinhard+CLAHE  | Paper CNN fusion (2022)            | SVM          | primary   |     0.9537 | 0.9713 |    0.9443 | 0.9723 |             0.4685 |
| otsu          | reinhard+CLAHE  | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.9551 | 0.9721 |    0.9457 | 0.9752 |             0.4685 |
| sam           | macenko+CLAHE   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9798 | 0.9738 |    0.9489 | 0.9972 |             0.9565 |
| sam           | macenko+CLAHE   | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9793 | 0.9732 |    0.9478 | 0.9983 |             0.9565 |
| sam           | macenko+CLAHE   | Paper CNN fusion (2022)            | SVM          | primary   |     0.9391 | 0.9225 |    0.8561 | 0.9827 |             0.9565 |
| sam           | macenko+CLAHE   | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.9378 | 0.9211 |    0.8538 | 0.9853 |             0.9565 |
| sam           | none            | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9803 | 0.9735 |    0.9483 | 0.9974 |             0.923  |
| sam           | none            | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9802 | 0.9733 |    0.9479 | 0.9984 |             0.923  |
| sam           | none            | Paper CNN fusion (2022)            | SVM          | primary   |     0.9586 | 0.9442 |    0.8943 | 0.9897 |             0.923  |
| sam           | none            | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.959  | 0.945  |    0.8957 | 0.9908 |             0.923  |
| sam           | reinhard+CLAHE  | Hybrid ConvNeXt+DINOv2+CBAM (ours) | SVM          | primary   |     0.9799 | 0.9749 |    0.951  | 0.9974 |             0.9176 |
| sam           | reinhard+CLAHE  | Hybrid ConvNeXt+DINOv2+CBAM (ours) | XGBoost      | primary   |     0.9801 | 0.9751 |    0.9514 | 0.9978 |             0.9176 |
| sam           | reinhard+CLAHE  | Paper CNN fusion (2022)            | SVM          | primary   |     0.9502 | 0.938  |    0.8833 | 0.9895 |             0.9176 |
| sam           | reinhard+CLAHE  | Paper CNN fusion (2022)            | XGBoost      | primary   |     0.9544 | 0.943  |    0.8922 | 0.9908 |             0.9176 |