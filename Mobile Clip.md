
## Maybe useful citations

[12]? Transformers for recognition at scale
[68] cross model affinity mimicking
[33,46] unimodal model ensembilnd; multimodal setup > utile per capire bene come hanno fatto l'ensemble
[74] IMPORTANTE: CoCa, il modello usato per fare le caption sintetiche
[62] fastvit e' da vedere come funziona
[61] forse dare uno sguardo
[18] (scaricato) Datacomp: In search of the next generation of multimodal datasets -- il dataset usato come base per creare il loro dataset rinforzato
[14] Reinforce Data
[47] (scaricato) Learning Transferable Visual Models From Natural Language Supervision -- standard multi-modal contrastive learning -- usato come esempio negativo: *at small-scale results in poor accuracies, which do not provide a useful signal to guide architecture design choices*.

Cos'e' model distillation?

[[xopp/2311.17049_mobile_clip.pdf|2311.17049_mobile_clip]]
OpenAI ViT-B/16 CLIP


---------------------
# Mobile CLIP
## In a nutshell
### Goal
- design a new family of aligned image-text encoders suitable for mobile devices.
### Needs
1. Tradeoff beween runtime performance and accuracy of different architectures.
	- large scale training of CLIP models is computationally expensive, so rapid devop and exploration of efficient architecture design is hard
	- standard multi-model contrastive learning [47] at small-scale results in poor accuracies, which doesn't provide a useful signal to guide architecture design choices
2. Reduced capacity of smaller architectures leads to subpar accuracy; can be improved with a better training method
### Overview - What they did 
A novel training approach based on the dataset reinforcement method [14] :
1. reinforce the dataset once with additional information
2. use the reinforced dataset several times for experimentation
Results in improved accuracy wrt original dataset.

Propose:
- multi-model variant of dataset reinforcement for training efficient CLIP models.
	- reinforced dataset
	- training architecture
- model
	- text encoder: *Text-RepMixer*: convolutional token mixer that decouples train-time and inference=time architectures.
	- image encoder: oved hybrid vision transformer called MCi based on the recent FastViT
## In detail
### Dataset
Reinforce [14] the image-text DataComp [18] adding sysnthetic captions and embedding from a strong ensemble of pretrained CLIP models (From table 12):
- teacher 1: `openai-ViT-L-14`
- teacher 2: `datacomp_xl_s13b_b90k-ViT-L-14`

Two variants of the datasets:
- DataCompDR-12M
- DataCompDR-1B
Shows significant learning efficiency improvement.

We introduce two variants of our reinforced datasets: DataCompDR-12M and DataCompDR-1B. Using Data- CompDR, we demonstrate 10x-1000x learning efficiency in comparison to DataComp.

Our proposed reinforced multi-modal dataset also benefits from synthetically generated captions, which we show are crucial for improved learning efficiency.


### Training Architecture: Multi-modal reinforced training
We introduce multi-modal reinforced training, a novel training strategy that incorporates knowledge transfer from a pre-trained image captioning model and an ensemble of strong CLIP models to improve learning efficiency.

Our proposed multi-modal reinforced training also includes cross-modal affinity mimicking [68 ]. Further, we extend uni- modal model ensembling [33, 46] to multimodal setup, and store targets obtained from an ensemble of CLIP models.

We extend the dataset reinforcement strategy [14 ] to the multi-modal setup of CLIP. 

Our proposed reinforced multi-modal datasets result in significant accuracy improvement without adding a training-time computational overhead.

-- 
training leverages knowledge transfer 
Two main components: 
1. leveraging the knowledge of an image captioning model via synthetic captions
2. knowledge distillation of image-text alignments from an ensemble of strong pretrained CLIP models. We follow the dataset reinforcement strategy of [ 14] and store the additional knowledge (synthetic captions and teacher embeddings) in the dataset (see Fig. 3)

#### Dataset Reinforcement
##### Synthetic captions - Aka Caption Augmentation
Motivation: image-text datasets used to train CLIP models are mostly sourced from the web, which is noisy. DataComp [18] and data filtering networks [16] improved the quality, but captions may not be descriptive enoiugh.

In order to boost the visual descriptiveness of caption we use CoCa [74] model and generate multiple synthetic captions for each image. $\rightarrow$ real captions in comparison to synthetic captions are generally more specific but noisier.

##### Image Augmentation
For each image $x_{img}^{(i)}$, we generate multiple augmented images $\hat{x}_{img}^{(i)}$ using a parmetrized augmentation function $\mathcal{A}$  
$$\hat{x}_{img}^{(i,j)} = \mathcal{A}(x_{img}^{(i)}; a^{(i,j)})$$
where $a^{(i,j)}$ are the augmentation parameters that are sufficient to reproduce $\hat{x}_{img}^{(i)}$ from $x_{img}^{(i)}$. 
The number and different kind of augmentations are provided in Tabs 4a and 13. 

Using coca_ViT-L-14 in OpenCLIP and strong random image augmentation
- DataCompDR-12M: 30
- DataCompDR-1B: 10
##### Ensemble Teacher
Model ensembling is used to create a stronger model froma set of indipendently trained ones [33,46]. We extend this technique to multi-model setup and use an ensemble of *K* CLIP models as a strong teacher.

We compute the feature embeddings of these models for augmented images $\hat{x}^{(i,j)}_{img}$ and synthetic captions $x^{(i,s)}_{syn}$ obtaining $d_k$-dimensional vectors $\psi^{(i,j,k)}_{img}$ and $\psi^{(i,s,k)}_{syn}$ for the $k_{th}$ teacher model. We also compute the teacher embeddings $\psi^{(i,k)}_{txt}$ of the ground-truth captions $x^{(i)}_{txt}$ (see Fig. 3b).

##### Reinforced Dataset
We store the image augmentation parameters a (i,j) , synthetic captions $x^{(i,s)}_{syn}$ , feature embeddings  $\psi^{(i,j,k)}_{img}$ , $\psi^{(i,s,k)}_{syn}$ and $\psi^{(i,k)}_{txt}$ of the CLIP teachers as additional knowledge in the dataset along with the original image $x^{(i)}_{img}$ and caption $x^{(i)}_{txt}$ (see Fig. 3c).
#### Training
##### Loss Function
take a look at the paper, too long to copy it here now.
##### Efficient Training
For every sample, we read the image $x_{img}^{(i)}$ and the corresponding ground-truth caption $x^{(i)}_{txt}$ form the dataset.
Ten we randomly load one of stored augmentation parameters $a^{(i,j)}$ and reproduce the augmented image $\hat{x}^{(i,j)}_{img}$ . We also randomly load one of synthetic captions $x_{syn}^{(i,s)}$. Finally we read the stored embeddings $\psi_{img}^{(i,j,k)}$, $\psi_{syn}^{(i,s,k)}$ and $\psi_{txt}^{(i,k)}$, corresponding to the *K* teacher models.
Using this data we construct two data batches:
- $\mathcal{B}_{real}$ augmented image, real caption pairs
- $\mathcal{B}_{syn}$ augmented image, synthetic caption pairs
and compute our trainig loss separately on both. The final loss is:
$$\sum_{\mathcal{B} \in \{\mathcal{B}_{real}, \mathcal{B}_{syn}\}} \mathcal{L}_{Total}(\mathcal{B})$$
Note that we can compute the total loss after a forward pass of the student model without any extra teacher related computations since the teacher embeddings required to compute the distillation loss are readily available as part of the dataset.
### Model Architecture
Using DataCompDR a new family of mobile-friendly aligned image-text encoders called MobileCLIP was developed, with a better latency-accuracy tradeoff compared to the previous works.

We design a new family of mobile-friendly CLIP models, MobileCLIP. Variants of MobileCLIP use hybrid CNN- transformer architectures with structural reparametrization in image and text encoders to reduce the size and latency. • We introduce multi-modal reinforced training, a novel training strategy that incorporates knowledge transfer from a pre-trained image captioning model and an ensemble of strong CLIP models to improve learning efficiency.

e introduce an improved convolution-transformer hybrid architecture for both vision and text modalities, that improve over recent state-of-the-art like [ 22, 38 , 44 , 53 ]. 

#### Text Encoder
- Classic CLIP is paired the vision transformer with a classical transformer with self-attention layers for text encoding; this works well but it's not efficient
- Recent work [67] showed that convolutions can be as effective for text encoding
- We found that purely convolutional architectures underperform their transformer counterparts
- We introduce a ***hybrid text encoder which makes use of 1-D convolutions and self-attention layers***: *Text-RepMixer* which decouples train-time and inference-time architectures.
- Inspured by reparametrizable convolutional token mixing (RepMixer, introduced in [62]). More in the paper and Appendix F. (It's probably quite complex lol)

This encoder is smaller, faster and has similar performance as the base text encoder paired with ViT-S/16.
#### Image Encoder
For Mobile-CLIP we introduce an improved hybrid vision transformer called MCi based on the recent FastViT [62].
To improve parameter efficiency we lower the expansion ration to 3.0 and increase the depth of the architecture. More in Appendix A.

##### Further Improvements
The optimizations introduced in [3 , 68 ] can be used to further improve efficiency of our models.
#### Architectural design techniques
##### Structural Reparametrization
9. TODO
10. TODO
11. TODO
21. TODO
61. TODO
##### Convolutional Token Mixing
62. TODO

