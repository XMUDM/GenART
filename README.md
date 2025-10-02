# GenART

This repository contains the pytorch implementation of **GenART: Advancing Genomic Foundation Models with Accurate Prediction Across Diverse Tasks and Adaptive Recognition of Functional DNA Elements**. 

GenART is a genomic foundation model pre-trained on 53 billion nucleotides from 15,844 species. GenART achieves state-of-the-art performance across 33 tasks—including histone modification profiling, enhancer and promoter prediction, splice site annotation, transcription factor binding, and virus variant classification. GenART surpasses previous leading models by an average of 4.34% in predictive accuracy, and improves histone modification classification by up to 14%. This strong performance stems from GenART’s novel architecture, which features an adaptive tokenization mechanism that dynamically segments raw DNA sequences into context-aware, variable-length tokens—learned during pretraining and refined through task-specific optimization. This design not only enhances representation learning but also enables the precise delineation of key genomic elements, such as CDS, exons, and UTRs, yielding a 72% improvement in Youden’s index over the prior best models.



### Setup

The following packages are required for running GenART. Compatibility is guaranteed with the specified versions:
- **numpy**: `1.24.4`
- **torch**: `1.13.1+cu117`
- **torchvision**: `0.14.1+cu117`
- **transformers**: `4.39.3`
- **flash-attn**: `2.3.6`

### Datasets

We provide the pre-training dataset, processed downstream task datasets, and pre-trained models. You can access them via the following Google Drive links:

[Pre-training dataset](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)

[NT benchmark](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)

[GUE benchamrk](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)

[Pre-trained models](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)


## Usage

We provide code for pretraining and finetuning, which can be found in `pretrain.py`, `finetune.py`, and `finetune_with_density_cluster.py`.
