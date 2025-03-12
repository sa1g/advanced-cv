# AdvancedCV project - MobileCLIP 

![MobileClip0](./images/MCDatasetR.png)

This is the repository contaning the material for the **Advanced Computer Vision project** (2024/2025)
## Table of Contents

  - [1. Overview](#1-overview)
  - [2. Possible Improvements](#2-possible-improvements)
    - [Training](#training)
    - [Inference](#inference)
  - [3. Project Material](#project-material)
    - [Presentation](#presentation)
    - [Relation](#relation)
  - [4. References](#4-references)
  - [Contributors](#contributors)

## 1. Overview

MobileCLIP is a family of **aligned image-text encoders** designed for **mobile devices**, focusing on **small size**, **low latency**, and **competitive accuracy**. Traditional CLIP models often require large-scale training and inference resources.

The innovations done are:
1. **Creating a Reinforced Dataset**, designed for training compact CLIP models using transfer learning techniques **(DataCompDR)**.
2. **Proposing a training approach** where the reinforced dataset comes into play to accelerate and guide the learning process.
3. **Developing hybrid CNN-transformer architectures** that use **structural reparametrization** to reduce model size and boost inference speed **(Text-RepMixer)**.  

These together reach **better latency-accuracy tradeoff** when compared to other state-of-the-art models.

## 2. Possible Improvements

### Training

- **TinyCLIP distillation of MobileCLIP**
  - Reduce the model parameter number, while keeping most of the original accuracy using TinyCLIP.
- **Improving synthetic captions**
  - Regenerate captions if they are too similar in the reinforced dataset.
- **Sigmoid self-attention**
  - Substitute softmax with sigmoid self-attention in the self-attention layers.
### Inference

- **PuMer adaptation to MobileCLIP**
  - PuMer adopts token pruning **(TIP)** and merging **(MAM)** in ViLT architecture to improve latency without compromising model performance.
  - MobileCLIP differs from ViLT from a structural perspective, but modality aware merging (MAM) still could lead to small latency improvements.
- **Token Ranking Pruning**
  - Patch Ranking original purpose was to **prune image patches** in tranformer based CLIP models **through a predictor** to reduce the number of tokens processed through the image and text encoder.
  - Adapting this technique to the MobileCLIP text encoder could result in similar improvements to the paper implementation.    
- **Dynamic Input Resolution**  
  - Adjust input resolution on a per-sample basis: low-resolution inference for simpler images, high-resolution for more complex ones.  

## 3. Project Material

<a name="presentation"></a>
**Presentation** of the project. - [link](./MobileClip_presentation.pdf) -

<a name="relation"></a>
**Relation** a detailed explanaition of our work. - [link](./MobileClip.md) -

## 4. References 
1. **MobileCLIP**
   - :scroll: [Paper](https://arxiv.org/abs/2311.17049v2) - *arXiv:2311.17049v2*
   - :computer: [GitHub](https://github.com/apple/ml-mobileclip)
2. **TinyCLIP**
   - :scroll: [Paper](https://arxiv.org/abs/2309.12314) - *arXiv:2309.12314*
   - :computer: [GitHub](https://github.com/wkcn/TinyCLIP?tab=readme-ov-file)
3. **PuMer**
   - :scroll: [Paper](https://arxiv.org/abs/2305.17530) - *arXiv:2305.17530*
   - :computer: [GitHub](https://github.com/csarron/PuMer)
4. **PatchRanking**
   - :scroll: [Paper](https://arxiv.org/html/2409.14607v1) - *arXiv:2409.14607v1* 
5. **Sigmoid self-attention**
   - :scroll: [Paper](https://arxiv.org/abs/2409.04431) - *arXiv:2409.04431*

## Contributors
- [Ettore Saggiorato](https://github.com/sa1g)

- [Emanuele Poiana](https://github.com/IlPoiana)

