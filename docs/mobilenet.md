
# MobileNet

## CIFAR100

|     arch    |  dataset | use_dcl | lambda | beta | gamma | swap_size |  top1  |  top5  |
|:-----------:|:--------:|:-------:|:------:|:----:|:-----:|:---------:|:------:|:------:|
| MobileNetV2 | CIFAR100 |  False  |    /   |   /  |   /   |     /     | 78.400 | 95.030 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  1.0  |   (7,7)   | 79.980 | 95.910 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  0.01 |  1.0  |   (7,7)   | 79.890 | 95.940 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  0.01  |   (7,7)   | 79.990 | 95.810 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  10  |   10  |   (7,7)   | 75.840 | 94.520 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  1.0  |   (5,5)   | 80.750 | 96.220 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  0.01  |   (5,5)   | 80.620 | 95.900 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  1.0  |   (3,3)   | 80.580 | 96.070 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  0.01  |   (3,3)   | 80.570 | 96.160 |

## CUB_200_2011

|     arch    |    dataset   | use_dcl | lambda | beta | gamma | swap_size |  top1  |  top5  |
|:-----------:|:------------:|:-------:|:------:|:----:|:-----:|:---------:|:------:|:------:|
| MobileNetV2 | CUB_200_2011 |  False  |    /   |   /  |   /   |     /     | 66.603 | 87.591 |
| MobileNetV2 | CUB_200_2011 |   True  |   1.0  |  1.0 |  1.0  |   (7,7)   | 68.467 | 89.662 |
| MobileNetV2 | CUB_200_2011 |   True  |   1.0  |  1.0 |  1.0  |   (5,5)   | 69.537 | 89.886 |
| MobileNetV2 | CUB_200_2011 |   True  |   1.0  |  1.0 |  1.0  |   (3,3)   | 70.193 | 90.024 |