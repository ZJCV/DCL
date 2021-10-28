<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/DCL.git"><img align="center" src="./imgs/DCL.png"></a></div>

<p align="center">
  «DCL»复现了论文<a title="" href="https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.html">Destruction and Construction Learning for Fine-Grained Image Recognition</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

* 解析：[ Destruction and Construction Learning for Fine-grained Image Recognition](https://blog.zhujian.life/posts/1f0dcd30.html)

更详细的训练数据可以查看：

* [Details](./docs/readme.md)

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

`DCL`设计了新的细粒度分类框架，通过联合训练分类网络和解构模块（`区域融合机制`和`对抗学习网络`）以及重构模块（`区域对齐网络`），实现了更好的性能增益，同时在推理时没有计算开销

当前实现基于[ JDAI-CV/DCL](https://github.com/JDAI-CV/DCL)。

## 安装

```
$ pip install -r requirements.txt
```

## 用法

* Train

```angular2html
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -cfg=configs/cub/r50_cub_448_e100_sgd_dcl_5x5_g4.yaml
```

* Test

```angular2html
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py -cfg=configs/cub/r50_cub_448_e100_sgd_dcl_5x5_g4.yaml
```

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

```
@InProceedings{Chen_2019_CVPR,
author = {Chen, Yue and Bai, Yalong and Zhang, Wei and Mei, Tao},
title = {Destruction and Construction Learning for Fine-Grained Image Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/DCL/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2021 zjykzj