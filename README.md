# IDvs.MoRec
This repository contains the source code for the paper ''Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited'', presented at SIGIR 2023 in the [paper](https://arxiv.org/pdf/2303.13835.pdf).

![](fig/IDvsMoRec.jpg) 

## Abstract
Recommendation models that utilize unique identities (IDs for short) to represent distinct users and items have been state-of-the-art (SOTA) and dominated the recommender systems (RS) literature for over a decade. Meanwhile, the pre-trained modality encoders, such as BERT and Vision Transformer, have become increasingly powerful in modeling the raw modality features of an item, such as text and images. Given this, a natural question arises: can a purely modality-based recommendation model (MoRec) outperforms or matches a pure ID-based model (IDRec) by replacing the itemID embedding with a SOTA modality encoder? In fact, this question was answered ten years ago when IDRec beats MoRec by a strong margin in both recommendation accuracy and efficiency.

We aim to revisit this `old' question and systematically study MoRec from several aspects. Specifically, we study several sub-questions: (i) which recommendation paradigm, MoRec or IDRec, performs better in practical scenarios, especially in the general setting and warm item scenarios where IDRec has a strong advantage? does this hold for items with different modality features? (ii) can the latest technical advances from other communities (i.e., natural language processing and computer vision) translate into accuracy improvement for MoRec? (iii) how to effectively utilize item modality representation, can we use it directly or do we have to adjust it with new data? (iv) are there any key challenges that MoRec needs to address in practical applications? To answer them, we conduct rigorous experiments for item recommendations with two popular modalities, i.e., text and vision. We provide the first empirical evidence that MoRec is already comparable to its IDRec counterpart with an expensive end-to-end training method, even for warm item recommendation. Our results potentially imply that the dominance of IDRec in the RS field may be greatly challenged in the future.

## Requirements
```
- torch == 1.8.0+cu111
- scikit-learn == 0.23.1
- numpy == 1.18.5
```


## Preparation

### Data Preparation



## Training
The training details are coming soon ...


## Details of the pre-trained ME

We report details of the pre-trained ME we used in Table.

| Pre-trained model | #Param. | URL |
| --- | --- | --- |
| BERT<sub>tiny</sub> | 4M  | https://huggingface.co/prajjwal1/bert-tiny |
| BERT<sub>small</sub> | 29M | https://huggingface.co/prajjwal1/bert-small |
| BERT<sub>base</sub> | 109M | https://huggingface.co/bert-base-uncased |
| RoBERTa<sub>base</sub> | 125M | https://huggingface.co/roberta-base |
| OPT<sub>125M</sub> | 125M | https://huggingface.co/facebook/opt-125M |
| ResNet18 | 12M | https://download.pytorch.org/models/resnet18-5c106cde.pth |
| ResNet34 | 22M | https://download.pytorch.org/models/resnet34-333f7ec4.pt |
| ResNet50 | 26M | https://download.pytorch.org/models/resnet50-19c8e357.pth |
| Swin-T | 28M | https://huggingface.co/microsoft/swin-tiny-patch4-window7-224 |
| Swin-B | 88M | https://huggingface.co/microsoft/swin-base-patch4-window7-224 |
| MAE<sub>base</sub> | 86M | https://huggingface.co/facebook/vit-mae-base |
