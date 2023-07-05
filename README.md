# Parts-aware-DINO
Official implementation of Parts based Attention for Highly Occluded Pedestrian Detection with Transformers

### Installation
Please refer to [base repository](https://github.com/IDEA-Research/DINO) for step-by-step installation. 

### Datasets Preparation and Evaluation scripts
Please refer to [Pedestron repository](https://github.com/hasanirtiza/Pedestron) for dataset preparation and evaluation scripts.

# Benchmarking 
### Benchmarking of our network on pedestrian detection datasets
| Dataset            | &#8595;Reasonable |  &#8595;Small   |  &#8595;Heavy   | 
|--------------------|:----------:|:--------:|:--------:|
| CityPersons        |  **8.69**   | **14.3** | **30.7** |  
| Caltech Pedestrian |  **2.2**   | **2.9**  | **32.92** |

## Training Details
* To ensure the correct directory is being used for training and validation data, adjustments need to be made in the [file](datasets/coco.py).
* The configuration file used for training can be found at [config file](config/DINO/DINO_4scale_swin.py). Citypersons dataset was trained using a batch size of 2, while Caltech was trained using a batch size of 4.
* During training, it is important to adjust the maximum image size as needed. Details can be seen in the [transform file](config/DINO/coco_transformer.py)

* The following command can be used for multi GPU training.
  ```shell 
  python tools/demo.py config checkpoint input_dir output_dir
  ```

# References

* [Pedestron](https://openaccess.thecvf.com/content/CVPR2021/papers/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.pdf)
* [DINO](https://arxiv.org/pdf/2203.03605.pdf)
