# MediSinGAN


This project aims to utlize SinGAN for tackling issues prevalent in the field of medical image analysis, especially image modality translation and synthetic data generation.
The project has been forked from [SinGAN](https://tamarott.github.io/SinGAN.htm)


# SinGAN

[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) | [Supplementary materials](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Shaham_SinGAN_Learning_a_ICCV_2019_supplemental.pdf) | [Talk (ICCV`19)](https://youtu.be/mdAcPe74tZI?t=3191) 
### Official pytorch implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image"
#### ICCV 2019 Best paper award (Marr prize)


## Random samples from a *single* image
With SinGAN, you can train a generative model from a single natural image, and then generate random samples from the given image, for example:

![](imgs/teaser.PNG)


## SinGAN's applications
SinGAN can be also used for a line of image manipulation tasks, for example:
 ![](imgs/manipulation.PNG)
This is done by injecting an image to the already trained model. See section 4 in the [paper](https://arxiv.org/pdf/1905.01164.pdf) for more details.


### Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{rottshaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Rott Shaham, Tamar and Dekel, Tali and Michaeli, Tomer},
  booktitle={Computer Vision (ICCV), IEEE International Conference on},
  year={2019}
}
```
<table>
  <tr>
    <td align="center"><a href="https://kentcdodds.com"><img src="https://avatars.githubusercontent.com/u/1500684?v=3?s=100" width="100px;" alt=""/><br /><sub><b>Kent C. Dodds</b></sub></a><br /><a href="#question-kentcdodds" title="Answering Questions">💬</a> <a href="https://github.com/all-contributors/all-contributors/commits?author=kentcdodds" title="Documentation">📖</a> <a href="https://github.com/all-contributors/all-contributors/pulls?q=is%3Apr+reviewed-by%3Akentcdodds" title="Reviewed Pull Requests">👀</a> <a href="#talk-kentcdodds" title="Talks">📢</a></td>
    <td align="center"><a href="https://github.com/jfmengels"><img src="https://avatars.githubusercontent.com/u/3869412?v=3?s=100" width="100px;" alt=""/><br /><sub><b>Jeroen Engels</b></sub></a><br /><a href="https://github.com/all-contributors/all-contributors/commits?author=jfmengels" title="Documentation">📖</a> <a href="https://github.com/all-contributors/all-contributors/pulls?q=is%3Apr+reviewed-by%3Ajfmengels" title="Reviewed Pull Requests">👀</a> <a href="#tool-jfmengels" title="Tools">🔧</a></td>
    <td align="center"><a href="https://jakebolam.com"><img src="https://avatars2.githubusercontent.com/u/3534236?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jake Bolam</b></sub></a><br /><a href="https://github.com/all-contributors/all-contributors/commits?author=jakebolam" title="Documentation">📖</a> <a href="#tool-jakebolam" title="Tools">🔧</a> <a href="#infra-jakebolam" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-jakebolam" title="Maintenance">🚧</a> <a href="https://github.com/all-contributors/all-contributors/pulls?q=is%3Apr+reviewed-by%3Ajakebolam" title="Reviewed Pull Requests">👀</a> <a href="#question-jakebolam" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://github.com/tbenning"><img src="https://avatars2.githubusercontent.com/u/7265547?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tyler Benning</b></sub></a><br /><a href="#maintenance-tbenning" title="Maintenance">🚧</a> <a href="https://github.com/all-contributors/all-contributors/commits?author=tbenning" title="Code">💻</a> <a href="#design-tbenning" title="Design">🎨</a></td>
    <td align="center"><a href="https://sinchang.me"><img src="https://avatars0.githubusercontent.com/u/3297859?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jeff Wen</b></sub></a><br /><a href="#maintenance-sinchang" title="Maintenance">🚧</a> <a href="https://github.com/all-contributors/all-contributors/pulls?q=is%3Apr+reviewed-by%3Asinchang" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="http://maxcubing.wordpress.com"><img src="https://avatars0.githubusercontent.com/u/8260834?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Maximilian Berkmann</b></sub></a><br /><a href="#translation-Berkmann18" title="Translation">🌍</a> <a href="https://github.com/all-contributors/all-contributors/commits?author=Berkmann18" title="Documentation">📖</a> <a href="#maintenance-Berkmann18" title="Maintenance">🚧</a> <a href="https://github.com/all-contributors/all-contributors/pulls?q=is%3Apr+reviewed-by%3ABerkmann18" title="Reviewed Pull Requests">👀</a> <a href="#talk-Berkmann18" title="Talks">📢</a></td>
    <td align="center"><a href="http://matheu.srv.br"><img src="https://avatars0.githubusercontent.com/u/23284276?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Matheus Rocha Vieira</b></sub></a><br /><a href="#translation-MatheusRV" title="Translation">🌍</a> <a href="https://github.com/all-contributors/all-contributors/commits?author=MatheusRV" title="Code">💻</a> <a href="https://github.com/all-contributors/all-contributors/commits?author=MatheusRV" title="Documentation">📖</a></td>
  </tr>
 </table>
