# IDvs.MoRec

### In-batch debiased cross-entropy loss

We find that use in-batch debiased cross-entropy loss can significantly enhance the performance of IDRec and MoRec compared with the binary cross entropy loss we used:

```math
-\sum\limits_{u \in \mathcal{U}} \sum\limits_{ i \in [2,...,L]}  \log \frac{\exp(\hat{y}_{ui} - \log(p_i))}{\exp(\hat{y}_{ui} - \log(p_i)) + \sum_{j \in [B], j \notin I_u} \exp(\hat{y}_{uj} - \log(p_j))}
```

where (p(i)) represents the popularity of item (i) in the dataset. We conducted experiments on in-batch debiased cross-entropy loss using SASRec and report the results:

| Dataset | Metrics | IDRec | BERT<sub>small</sub> | BERT<sub>base</sub> | RoBERTa<sub>base</sub> | Improv. |
| --- | --- | --- | --- | --- | --- | --- |
| MIND | HR@10 | 22.60 | 22.96 | 22.82 | **23.00** | +1.77% |
| MIND | NDCG@10 | 12.57 | **12.82** | 12.70 | **12.82** | +1.99% |
| **Dataset** | **Metrics** | **IDRec** | **ResNet50** | **Swin-T** | **Swin-B** | **Improv.** |
| HM  | HR@10 | 11.94 | 11.90 | 12.20 | **12.26** | +2.68% |
| HM  | NDCG@10 | **7.75** | 7.46 | 7.70 | 7.70 | -0.65% |
| Bili | HR@10 | 4.91 | 5.62 | 5.55 | **5.73** | +16.70% |
| Bili | NDCG@10 | 2.71 | 3.08 | 3.03 | **3.14** | +15.87% |

### Details of the pre-trained ME

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
