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

## Contributors ✨

Our wonderful people  💻 :

* [Amrit Kumar Jethi](https://github.com/amritkumar9595)
* [Rajkumar Vaghashiya](https://github.com/rvaghashiya)
* [Екатерина Неповинных](https://github.com/kwadraterry)
* 
## LICENSE

[MIT](LICENSE)
