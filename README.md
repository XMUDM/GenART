# GenART

This repository contains the pytorch implementation of **GenART: Advancing Genomic Foundation Models with Accurate Prediction Across Diverse Tasks and Adaptive Recognition of Functional DNA Elements**. 

GenART is a genomic foundation model pre-trained on 53 billion nucleotides from 15,844 species. GenART achieves state-of-the-art performance across 33 tasks—including histone modification profiling, enhancer and promoter prediction, splice site annotation, transcription factor binding, and virus variant classification. GenART surpasses previous leading models by an average of 4.34% in predictive accuracy, and improves histone modification classification by up to 14%. This strong performance stems from GenART’s novel architecture, which features an adaptive tokenization mechanism that dynamically segments raw DNA sequences into context-aware, variable-length tokens—learned during pretraining and refined through task-specific optimization. This design not only enhances representation learning but also enables the precise delineation of key genomic elements, such as CDS, exons, and UTRs, yielding a 72% improvement in Youden’s index over the prior best models.

![Schematic illustration of the GenART framework.](GenART.png)

**Schematic illustration of the GenART framework.** **(a)** The overall model architecture and pre-training-fine-tuning framework of the GenART model. **(b)** A detailed schematic of the Adaptive Tokenization module. **(c)** Downstream applications and performance evaluation of genomic foundation models.

### Setup

The following packages are required for running GenART. Compatibility is guaranteed with the specified versions:
- **numpy**: `1.24.4`
- **torch**: `1.13.1+cu117`
- **torchvision**: `0.14.1+cu117`
- **transformers**: `4.39.3`
- **flash-attn**: `2.3.6`

### Datasets

We provide NT benchmark datasets, GUE benchmark datasets and checkpoints for pretrained models. You can access them via the following Google Drive links:

[NT benchmark](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)

[GUE benchamrk](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)

[Pre-trained GenART-350M](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)

[Pre-trained GenART-1B](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)


## Usage

We provide code for pre-training and fine-tuning, which can be found in `pretrain.py`, `finetune.py`, and `finetune_with_density_cluster.py`. `pretrain.py` is used for pre-training, `finetune.py` is used for fine-tuning, and `finetune_with_cluster_density.py` is used for refinement based on the adjustable hyperparameters *token density* and *cluster factor*.  
The execution commands are as follows:

```shell
# 1. Multi-GPU Pretraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 pretrain.py \
    --train_data /path/to/train_data.csv \
    --eval_data /path/to/test_data.csv \
    --output_dir /path/to/output_dir \
    --batch_size 8 \
    --max_steps 10000 \
    --save_steps 10000

# 2. Single-GPU Pretraining
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    --train_data /path/to/train_data.csv \
    --eval_data /path/to/test_data.csv \
    --output_dir /path/to/output_dir \
    --batch_size 8 \
    --max_steps 10000 \
    --save_steps 10000  

# 3. Multi-GPU Finetuning  
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 finetune.py \
    --train_data /path/to/train_data.csv \
    --eval_data /path/to/test_data.csv \
    --output_dir /path/to/output_dir \
    --pretrained_model /path/to/pretrained_model \
    --batch_size 8 \
    --max_steps 0 \
    --epochs 20

# 4. Single-GPU Finetuning  
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data /path/to/train_data.csv \
    --eval_data /path/to/test_data.csv \
    --output_dir /path/to/output_dir \
    --pretrained_model /path/to/pretrained_model \
    --batch_size 32 \
    --max_steps 0 \
    --epochs 20

# 5. Multi-GPU Finetuning with *token_density* and *cluster_factor*
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 finetune.py \
    --train_data /path/to/train_data.csv \
    --eval_data /path/to/test_data.csv \
    --output_dir /path/to/output_dir \
    --pretrained_model /path/to/pretrained_model \
    --batch_size 8 \
    --max_steps 0 \
    --epochs 20 \
    --token_density 0.7 \
    --cluster_factor 0.7

# 6. Single-GPU Finetuning with *token_density* and *cluster_factor*
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --train_data /path/to/train_data.csv \
    --eval_data /path/to/test_data.csv \
    --output_dir /path/to/output_dir \
    --pretrained_model /path/to/pretrained_model \
    --batch_size 32 \
    --max_steps 0 \
    --epochs 20 \
    --token_density 0.7 \
    --cluster_factor 0.7
```
