
# README

## CIFAR100

|     arch    |  dataset | use_dcl | lambda | beta | gamma | swap_size |  top1  |  top5  |
|:-----------:|:--------:|:-------:|:------:|:----:|:-----:|:---------:|:------:|:------:|
| MobileNetV2 | CIFAR100 |  False  |    /   |   /  |   /   |     /     | 78.400 | 95.030 |
| MobileNetV2 | CIFAR100 |   True  |   1.0  |  1.0 |  1.0  |   (5,5)   | 80.750 | 96.220 |
| ResNet50 | CIFAR100 |  False  |    /   |   /  |   /   |     /     | 82.000 | 96.270 |
| ResNet50 | CIFAR100 |   True  |   1.0  |  1.0 |  0.01  |   (3,3)   | 82.980 | 96.720 |

## CUB_200_2011

* `224x224`

|     arch    |    dataset   | use_dcl | lambda | beta | gamma | swap_size |  top1  |  top5  |
|:-----------:|:------------:|:-------:|:------:|:----:|:-----:|:---------:|:------:|:------:|
| MobileNetV2 | CUB_200_2011 |  False  |    /   |   /  |   /   |     /     | 66.603 | 87.591 |
| MobileNetV2 | CUB_200_2011 |   True  |   1.0  |  1.0 |  1.0  |   (3,3)   | 70.193 | 90.024 |
| ResNet50 | CUB_200_2011 |  False  |    /   |   /  |   /   |     /     | 69.330 | 89.489 |
| ResNet50 | CUB_200_2011 |   True  |   1.0  |  1.0 |  1.0  |   (3,3)   | 71.936 | 90.956 |

* `448x448`

|   arch   |    dataset   | use_dcl | lambda | beta | gamma | swap_size |  top1  |  top5  |
|:--------:|:------------:|:-------:|:------:|:----:|:-----:|:---------:|:------:|:------:|
| ResNet50 | CUB_200_2011 |  False  |    /   |   /  |   /   |     /     | 81.315 | 95.100 |
| ResNet50 | CUB_200_2011 |   True  |   1.0  |  0.1 |  1.0  |   (5,5)   | 83.575 | 95.997 |

## See

* [MobileNet](./mobilenet.md)
* [ResNet](./resnet.md)