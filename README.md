# IDvs.MoRec
This repository contains the source code for the **SIGIR 2023** paper **''Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited''**.

Full version in [[PDF]](https://arxiv.org/pdf/2303.13835.pdf).

![](fig/IDvsMoRec.jpg) 

## Abstract
Recommendation models that utilize unique identities (IDs for short) to represent distinct users and items have been state-of-the-art (SOTA) and dominated the recommender systems (RS) literature for over a decade. Meanwhile, the pre-trained modality encoders, such as BERT and Vision Transformer, have become increasingly powerful in modeling the raw modality features of an item, such as text and images. Given this, a natural question arises: can a purely modality-based recommendation model (MoRec) outperforms or matches a pure ID-based model (IDRec) by replacing the itemID embedding with a SOTA modality encoder? In fact, this question was answered ten years ago when IDRec beats MoRec by a strong margin in both recommendation accuracy and efficiency.

We aim to revisit this `old' question and systematically study MoRec from several aspects. Specifically, we study several sub-questions: (i) which recommendation paradigm, MoRec or IDRec, performs better in practical scenarios, especially in the general setting and warm item scenarios where IDRec has a strong advantage? does this hold for items with different modality features? (ii) can the latest technical advances from other communities (i.e., natural language processing and computer vision) translate into accuracy improvement for MoRec? (iii) how to effectively utilize item modality representation, can we use it directly or do we have to adjust it with new data? (iv) are there any key challenges that MoRec needs to address in practical applications? To answer them, we conduct rigorous experiments for item recommendations with two popular modalities, i.e., text and vision. We provide the first empirical evidence that MoRec is already comparable to its IDRec counterpart with an expensive end-to-end training method, even for warm item recommendation. Our results potentially imply that the dominance of IDRec in the RS field may be greatly challenged in the future.

## Requirements
```
- torch == 1.7.1+cu110
- torchvision==0.8.2+cu110
- transformers==4.20.1
```


## Preparation

### Data Download 
The complete news recommendation dataset (MIND) is visible under the `dataset/MIND`, and the dataset with vision (HM and Bili) requires the following actions:

Download the image file "hm_images.zip" (100,000 images in 3x224x224 size) for Hm dataset from this [link](https://drive.google.com/file/d/1zm0V3th-_ZxAevQM5yt8tkbLHnXGc6lk/view?usp=share_link). 

Unzip the downloaded model file `hm_images.zip`, then put the unzipped directory `hm_images` into `dataset/Hm/` for the further processing.

**Mentions:**
The Bili dataset we used is from an unpublished paper, temporarily available via email (yuanfajie@westlake.edu.cn, lifengyi@westlake.edu.cn, yuanzheng@westlake.edu.cn). Please fill out the applicant form `The Usage Guidelines of Bili.pdf` and provide your name and affiliation information (using official email) when requesting the dataset via email. Please send the same application email to the 3 email addresses mentioned above.


### Data Preparation
You need to process the images file of HM dataset to a LMDB database for efficient loading during training.

```
cd dataset/HM
python run_lmdb_hm.py
```

### Pre-trained Model Download

We report details of the pre-trained ME we used in Table. Download the pytorch-version of them, and put the checkpoint `pytorch_model.bin` into the corresponding path under `pretrained_models/`

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

## Training
An example:
For training text MoRec with SASRec in **end2end** manner, and using **bert-base** as the modality encoder:
```
cd bce_text/main-end2end
python train_bert_base.py
```
After training, you will get the checkpoint of the MoRec model, then set the parameters in  `test_bert_base.py` and run it for the test result.

**Mentions:**
You can change the `train_xxx.py` and the `test_xxx.py` to set the hyperparameters.
The recommended GPU resource can be found in Table 6 in the paper.


## Citation
If you use our code or find IDvs.MoRec useful in your work, please cite our paper as:

```bib
@article{yuan2023go,
  title={Where to Go Next for Recommender Systems? ID-vs. Modality-based recommender models revisited},
  author={Yuan, Zheng and Yuan, Fajie and Song, Yu and Li, Youhua and Fu, Junchen and Yang, Fei and Pan, Yunzhu and Ni, Yongxin},
  journal={arXiv preprint arXiv:2303.13835},
  year={2023}
}
```


