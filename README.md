# Interpretation Before Integration: LLM-Guided Multi-Modal Completion and Fusion Network for Survival Analysis with Incomplete Data

This repository is the official implementation of *LMCF*. If you encounter any question, please feel free to contact us. You can create an issue. Also welcome for any idea exchange and discussion.

## Updates

[**26/06/2025**] Conducting code cleaning. Waiting to be made public.



## Introduction


Multi-modality survival analysis for nasopharyngeal carcinoma (NPC) holds great potential for improving prognosis prediction and clinical decision-making. However, it is challenged by structural and semantic misalignments across heterogeneous data. Structural misalignment arises from incomplete clinical records, where missing data introduce uncertainty in prediction. Semantic misalignment stems from the gap between structured modalities (e.g., clinical and radiomic features) and unstructured data like 3D MRI, hindering effective feature integration. Existing methods often ignore missing data or compress multi-modal information into scalar representations, failing to capture complex modality interactions and solve the problem of semantic misalignment. Furthermore, current completion techniques typically lack interpretability and overlook joint modeling of inter- and intra-sample correlations when dealing with structural misalignment, limiting their reliability in clinical settings. These issues are further exacerbated by over-parameterized models prone to overfitting in small-sample scenarios. To address these challenges, we propose LMCF, a Large Language Model Guided Multi-Modal Completion and Fusion (LMCF) Network tailored for survival analysis with Incomplete Data. LMCF consists of two core components: a Lightweight Dual-Branch Multi-Modality Enhanced Feature Encoding (LDME) Layer, which incorporates an Interpretable Multi-Source Cross-Modality Completer (IMCC) for explainable reconstruction of missing data to resolve structural misalignment; and an LLM-Guided Structure-Semantic Two-Stream Fusion (LSTF) Layer, equipped with a Quaternion Convolution-based Cross-domain Adaptive Attention Fusioner (QCAAF) to effectively integrate features across modalities and mitigate semantic misalignment. Extensive experiments on The Cancer Genome Atlas (TCGA) and two proprietary NPC datasets (PRNN and NCD) from Sun Yat-sen University Cancer Center demonstrate LMCF’s superior performance in survival prediction and risk stratification, particularly under conditions of incomplete modalities and limited data resources.


## Train & Eval

Public datasets can be downloaded from the [GDC](https://portal.gdc.cancer.gov/). 

```
cd LMCF_code
python train.py
```

After modifying the dataset path and setting the parameters in the file train.py, you can directly use the command above line for training. The training process will be printed out, and the prediction results will be saved in the path. Due to equipment failure, we haven't finished the code cleaning and datasets cleaning yet, so the code now is not completed.

