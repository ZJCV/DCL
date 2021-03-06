<div align="right">
  è¯­è¨:
    ð¨ð³
  <a title="è±è¯­" href="./README.md">ðºð¸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/DCL.git"><img align="center" src="./imgs/DCL.png"></a></div>

<p align="center">
  Â«DCLÂ»å¤ç°äºè®ºæ<a title="" href="https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.html">Destruction and Construction Learning for Fine-Grained Image Recognition</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

* è§£æï¼[ Destruction and Construction Learning for Fine-grained Image Recognition](https://blog.zhujian.life/posts/1f0dcd30.html)

æ´è¯¦ç»çè®­ç»æ°æ®å¯ä»¥æ¥çï¼

* [Details](./docs/readme.md)

## åå®¹åè¡¨

- [åå®¹åè¡¨](#åå®¹åè¡¨)
- [èæ¯](#èæ¯)
- [å®è£](#å®è£)
- [ç¨æ³](#ç¨æ³)
- [ä¸»è¦ç»´æ¤äººå](#ä¸»è¦ç»´æ¤äººå)
- [è´è°¢](#è´è°¢)
- [åä¸è´¡ç®æ¹å¼](#åä¸è´¡ç®æ¹å¼)
- [è®¸å¯è¯](#è®¸å¯è¯)

## èæ¯

`DCL`è®¾è®¡äºæ°çç»ç²åº¦åç±»æ¡æ¶ï¼éè¿èåè®­ç»åç±»ç½ç»åè§£ææ¨¡åï¼`åºåèåæºå¶`å`å¯¹æå­¦ä¹ ç½ç»`ï¼ä»¥åéææ¨¡åï¼`åºåå¯¹é½ç½ç»`ï¼ï¼å®ç°äºæ´å¥½çæ§è½å¢çï¼åæ¶å¨æ¨çæ¶æ²¡æè®¡ç®å¼é

å½åå®ç°åºäº[ JDAI-CV/DCL](https://github.com/JDAI-CV/DCL)ã

## å®è£

```
$ pip install -r requirements.txt
```

## ç¨æ³

* Train

```angular2html
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -cfg=configs/cub/r50_cub_448_e100_sgd_dcl_5x5_g4.yaml
```

* Test

```angular2html
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py -cfg=configs/cub/r50_cub_448_e100_sgd_dcl_5x5_g4.yaml
```

## ä¸»è¦ç»´æ¤äººå

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## è´è°¢

```
@InProceedings{Chen_2019_CVPR,
author = {Chen, Yue and Bai, Yalong and Zhang, Wei and Mei, Tao},
title = {Destruction and Construction Learning for Fine-Grained Image Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## åä¸è´¡ç®æ¹å¼

æ¬¢è¿ä»»ä½äººçåä¸ï¼æå¼[issue](https://github.com/ZJCV/DCL/issues)ææäº¤åå¹¶è¯·æ±ã

æ³¨æ:

* `GIT`æäº¤ï¼è¯·éµå®[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)è§è
* è¯­ä¹çæ¬åï¼è¯·éµå®[Semantic Versioning 2.0.0](https://semver.org)è§è
* `README`ç¼åï¼è¯·éµå®[standard-readme](https://github.com/RichardLitt/standard-readme)è§è

## è®¸å¯è¯

[Apache License 2.0](LICENSE) Â© 2021 zjykzj