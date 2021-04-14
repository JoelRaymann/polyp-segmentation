# GAR-Net: Guided Attention Residual Network for Polyp Segmentation from Colonoscopic Video Frames

GAR-Net is a Deep Learning model developed from segmenting Polyps from Colonoscopy Video Frames and images. 

## Requirements
### Environment Supported
 - Windows 8, 8.1 and 10 (Preferred)
 - MacOS (High Sierra or Higher)
 - Any Debian or RPM based Linux OS

### Software Requirements
 - Python version 3.6.8 or higher.
 - Nvidia CUDA version 10.1 or higher (For GPU Training and Inference)

### Preferred Hardware Requirements
 - Any 4 core processor
 - 16GB RAM or Higher
 - Nvidia GPU (Required for training and inferencing with GPU) 
  
## Training Steps
Currently 8 models are supported out of the box for training and testing. Any of these models can be trained on the supported CVC-ClinicDB dataset or the Kvasir-SEG dataset. The supported models are:
```yaml
1. UNet-Dice: U-Net model.
2. UNet-Attn: U-Net with Attention Gated model.
3. ResUNet: Residual block based U-Net.
4. DeepLabv3: DeepLabv3 model.
5. FCN8: Fully connected Convolution Neural Network model.
6. SegNet: SegNet model.
7. UNet-GuidedAttn: U-Net with GAM and GAL based Attention.
8. GAR-Net: The proposed GAR-Net model.
```

To train a model. Please do the following steps:

1. Setup python environment and install the required packages/dependencies with the provided `requirements.txt` file.
```sh
pip install -r "./requirements.txt"
```
2.  Following the setup, download the latest CVC-ClinicDB dataset or the Kvasir-SEG dataset.
3.  Edit the `train_config.yaml` file by mentioning the model_name (the model to train), the dataset_family (either "CVC-ClinicDB" or "Kvasir-Seg"), the dataset_path (the root path of the dataset), the batch sizes for train, valid and test and all other meta data.
4.  Then, run `train_model.py` script
```sh
python train_model.py -C "./train_config.yaml"
```
5. The model and the train logs are saved in `./outputs/` folder as weights.

## To visualize the outputs from the trained models
To visualize each sample in the CVC-ClinicDB or the Kvasir-SEG dataset. Please follow the steps belows:

1. Train the model if not trained. Follow the training steps.
2. Edit the `predict_config.yaml` file and edit the model_name (the model to take inference), the dataset_path, the dataset_family, the attn_output (set it true for models with attention mechanisms in it, else keep it as false) and all other meta data.
3. Then, run `predict_model.py` script
```sh
python predict_model.py -C "./predict_config.yaml"
```
4. The output inferences are stored as images in `./outputs/output_visualization/` folder

# Results
## Result Table for CVC-ClinicDB dataset
Models trained on CVC-ClinicDB dataset with batch-size = 4 and learning rate = 0.0001 over 120 epochs.
  
| S. No | Model Name | Dice Index | mIoU     | pixel Acc. |
| ----- | ---------- | ---------- | -------- | ---------- |
| 1     | FCN8       | 0.8724981  | 0.74308  | 0.9757281  |
| 2     | SegNet     | 0.73168665 | 0.811956 | 0.97676927 |
| 3     | UNet       | 0.8803434  | 0.767874 | 0.97531813 |
| 4     | UNet-Attn  | 0.89131886 | 0.783651 | 0.97713906 |
| 5     | ResUNet    | 0.89080334 | 0.781462 | 0.97680587 |
| 6     | DeepLabv3  | 0.90001955 | 0.819477 | 0.9817561  |
| 7     | GAR-Net    | 0.9100929  | 0.831234 | 0.9831491  |

## Result Table for Kvasir-SEG dataset
Models trained on Kvasir-SEG dataset with batch-size = 4 and learning rate = 0.0001 over 120 epochs.

| S. No | Model Name | Dice Index | mIoU     | pixel Acc. |
| ----- | ---------- | ---------- | -------- | ---------- |
| 1     | FCN8       | 0.73635364 | 0.548514 | 0.9176181  |
| 2     | SegNet     | 0.78634053 | 0.784731 | 0.9607266  |
| 3     | UNet       | 0.85862905 | 0.750284 | 0.959373   |
| 4     | UNet-Attn  | 0.8637741  | 0.754757 | 0.9587085  |
| 5     | ResUNet    | 0.86858636 | 0.761282 | 0.9630264  |
| 6     | DeepLabv3  | 0.87866044 | 0.805872 | 0.9693878  |
| 7     | GAR-Net    | 0.8915458  | 0.815802 | 0.9717203  |


# TODO
- [ ] Add predict for single images
- [ ] Add predict for full videos