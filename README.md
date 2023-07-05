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
  python3 -m torch.distributed.launch --nproc_per_node= gpu_count --master_port=11001 main.py --output_dir path/to/output -c path/to/config --coco_path /path/to/dataset/ --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=/path/to/backbone/ --pretrain_model_path /path/to/pretrain_model --initilize_cross_attention --finetune_ignore label_enc.weight class_embed 
  ```
  * pretrained models can be found in the [link]()

# References

* [Pedestron](https://openaccess.thecvf.com/content/CVPR2021/papers/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.pdf)
* [DINO](https://arxiv.org/pdf/2203.03605.pdf)
