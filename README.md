<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/DCL.git"><img align="center" src="./imgs/DCL.png"></a></div>

<p align="center">
  «DCL» re-implements the paper <a title="" href="https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.html">Destruction and Construction Learning for Fine-Grained Image Recognition</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

Differ with other attention-based or part-based fine-classification methods, DCL adds an `Adversarial Loss` and `Region Align Network` in training, and only use backbone network in infer. Improve the accuracy of the model without affecting the reasoning speed.

Current project implementation is based on [ JDAI-CV/DCL](https://github.com/JDAI-CV/DCL).

## Installation

```
$ pip install -r requirements.txt
```

## Usage

...

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

```
@InProceedings{Chen_2019_CVPR,
author = {Chen, Yue and Bai, Yalong and Zhang, Wei and Mei, Tao},
title = {Destruction and Construction Learning for Fine-Grained Image Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/ZJCV/DCL/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj