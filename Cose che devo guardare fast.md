# TODO TOGETHER (ALONE at HOME)
1. Rileggere Mobile Clip
	1. obiettivi
	2. intro a cosa hanno introdotto
		1. reinforce data
		2. multi-model reinforced training (loss)
		3. hybrid text encoder (Text-Rep Mixer)
		4. hybrid image encoder (Fast Vit Inspired)
	3. risultati e specifiche (hw, latency, dataset, architettura, training time, storage size)
2. Possibili migliorie
	1. dataset reinforce
		1. https://arxiv.org/abs/2405.11919 -> idea di ettore, dataset caption intelligent reduction wrt estimated quality wrt generated captions -> objective to reduce the training time and potentially (small gain) in accuracy.
		2. PruMer (towards the dataset creation)-> 
			1. velocizzare la creazione del dataset rinforzato
	2. loss 
		1. implementare tinyClip -> obiettivo di ridurre ancora piu' il modello -> less latency, less training time
		2. Mostrare comparison tra risultati Mobile e Tiny: accuracy, dimensioni modello, training time
		3. Possibile implementazione di Patch Ranking per fare pruning sul ViT come sostituto alla sparsity(idea)
	3. inference
		1. prumer (towards model efficiency)
			1. rendere il modello piu' veloce (forse occupa meno vram? da vedere)
	4. architecture (?)
		1. Sigmoid Self-Attention -> NotebookLM salvaci tu

## 2 Possibili Migliorie

### 2.1 Dataset Reinforcement

### 2.2 TinyCLIP on MobileClip - Loss integration

**2.2.1 CLIP & MobileCLIP**

We know that std. **CLIP loss** is:

$$
\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{N} \Bigg[
\log \frac{\exp\bigl(\mathrm{sim}(x_{i}, y_{i}) / \tau\bigr)}
{\sum_{j=1}^N \exp\bigl(\mathrm{sim}(x_{i}, y_{j}) / \tau\bigr)}
\;+\;
\log \frac{\exp\bigl(\mathrm{sim}(x_{i}, y_{i}) / \tau\bigr)}
{\sum_{j=1}^N \exp\bigl(\mathrm{sim}(x_{j}, y_{i}) / \tau\bigr)}
\Bigg],

$$

Where is in the first contrastive loss from text to image and the second is viceversa

MobileClip loss is:

$$

\mathcal{L}_{\text{Total}}(\mathcal{B}) 
= (1 - \lambda)\,\mathcal{L}_{\text{CLIP}}(\mathcal{B})
  + \lambda\,\mathcal{L}_{\text{Distill}}(\mathcal{B}),\\

\mathcal{L}_{\text{Distill}}(\mathcal{B})
= \tfrac{1}{2}\,\mathcal{L}_{\text{I2T Distill}}(\mathcal{B})
  + \tfrac{1}{2}\,\mathcal{L}_{\text{T2I Distill}}(\mathcal{B}),\\

\mathcal{L}_{\text{I2T Distill}}(\mathcal{B})
= \frac{1}{bK}
  \sum_{k=1}^{K}
  \mathrm{KL}\Bigl(
      S_{\tau_k}\bigl(\Psi_{\mathrm{img}}^{(k)}, \Psi_{\mathrm{txt}}^{(k)}\bigr)
      \,\Big\|\,
      S_{\hat{\tau}}\bigl(\Phi_{\mathrm{img}}, \Phi_{\mathrm{txt}}\bigr)
  \Bigr).
$$

Which has the std. CLIP loss and the Distill loss parametrized by $\lambda$ which empowers the stored embeddings a lot (0.7-1.0), as shown in the figure below

![lambdaMobile](./images/lambda_mobileClip.png)

**TinyClip contributions**

**2.2.2 Affinity Mimicking**
This is the technique (introduced by a loss) that mixed the embeddings from the teacher and the student model, using the classic contrastive loss between both the Text2Image and viceversa among the teacher and the student

$$
L_{distill} = L_{I2T} + L_{T2I}\\
L_{I2T} = CE(A^{s}_{I2T}, A^{t}_{I2T})\\
A_{I2T}(i,j) = \frac{exp(I_{i} * T_{j}/ \tau)} {\sum_{k \epsilon \Beta}exp(I_{i} * T_{k}\tau)}
$$

**2.2.3 Weight Inheritance - Manual and Automatic**
*"To capture the importance of weights in a fine-grained level, we introduce two mask variables $m_{head}$ , $m_{int}$ ∈ {0, 1} to identify the redundant attention heads in MHA and neurons in FFN respectively, while keeping the important ones. These two kinds of masks are imposed on the activation of attention heads and the intermediate layer of FFN"*

1. $m_{head}^{h}$ *with* $h = (1 .. N_{H})$
2. $m_{int}$ *one for each FFN*
   
*"Moreover, to further learn the importance of embedding dimensions in transformer, we introduce an additional mask m embed ∈ {0, 1}. This mask is shared across all layers because each dimension in the hidden representation is connected to the corresponding dimension in the subsequent layer through a residual connection."*

1. $m_{embed} \epsilon [0,1]$ 

This three masks are learnt by the model, introducing them in the Loss in this way:

$$
L = L_{distill} + L_{sparsity}\\
L_{sparsity} = \lambda * (p - q) + \beta * (p - q)^2
$$

*"p is the overall compression rate of learnable masks for the model, including image encoder and text encoder"*
Very important is the *p* parameter which has to be equal to *q*, this because q is set manually to control the compression ratio :


![p_tinyClip](./images/p_tinyClip.png)

**2.2.4 Progressive Multi-Stage Distillation**
*"When attempting to achieve a high target sparsity, i.e.,>70%, compressing the model in a single stage can lead toa significant reduction in accuracy and even result in con-vergence failure. This is due to the fact that most weights ofthe large model are directly discarded, including those thatare important for ensuring model quality and convergence.As a solution, we propose a multi-stage progressive distil-lation method to achieve a high compression rate withoutseriously sacrificing accuracy. In each stage, we use a mod-est degree of compression, e.g., 25%, to avoid large loss ofperformance and make training stable."*

Just using the two precedent methods gradually, maybe changing also the percentage of compression. 

**2.2.5 Loss integration**
MobileClip distillation into TinyClip is a possibility, so two plausible Losses might be:

1. TinyClip
   $$
   L = L_{distill} + L_{sparsity}\\
   $$
2. Mixture of TinyClip and MobileClip
$$
\mathcal{L}_{\text{Total}}(\mathcal{B}) 
= (1 - \lambda)\,\mathcal{L}_{\text{CLIP}}(\mathcal{B})
  + \lambda((1 - \alpha)\mathcal{L}_{\text{Distill}}(\mathcal{B}) + \alpha\mathcal{L}_{\text{Sparsity}}(\mathcal{B}))   
$$

where:
- $\lambda$ balanced as usual the std. CLIP loss and the distillation one
- $\alpha$ instead balance the contribution of distill and sparsity
- $\mathcal{L}_{\text{Distill}}$ could be the MobileClip or the TinyClip loss, in both cases using the stored embeddings or the MobileClip embeddings
- $\mathcal{L}_{\text{Sparsity}}$ empowers the compression of the transformer architecture, eventually applied only to the self-attention layers in case of the hybrid transformer encoder and to the ConvFFN, FOR THIS APPROACH WE NEED TO KNOW THE WEIGHTS OF THE TEACHER MODEL, Which we don't have those in mobileClip standard training. -> **we need to distill even further mobileClip.**






## 2.3 Inference - PuMer
*Pumer Contributions*
### 2.3.1 TIP and MAM 
Inside a transformer architecture self-attention layers, we can use the keys of each token vector as a metric to understand the importance of each token. 

*"Each token reducer consists of two sequential non-parametric modules: first, a text-informed pruner (TIP) prunes image tokens that are not related to the accompanying text (Section 4.1); second, a modality-aware merger (MAM) reduces tokens by merging similar tokens within the image or text modality"*

**TIP:** taken as it's presented in the paper is not usable, cause CLIP doesn't have a shared transformer block layer where image and text tokens are mixed and consequently passed in a MHA layer.

**MAM:** We take the dot product between the keys of each token vector and the top scoring(the most similar) and merge them according to the merge ratio $\mathcal{r}$

$$
\mathcal{S_{p}^{t_1 t_2}} = \mathcal{K_{t_1}}\mathcal{K_{t_2}}
$$

with:
- $V$ image(text) token vectors
- $\mathcal{p}$ from [0 .. $\mathcal{k}^{'}$] where $k^{'}=(1 - k)|V|$ that is the number of kept token

### 2.4 Token Pruning through Patch Ranking for CLIP

*Published on: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, 2024.*

*"In this work, we introduce a novel framework designed forpruning patch tokens in CLIP’s ViT, effectively addressing the computational intensity typically associated with thesemodels. At the heart of our approach is **the “Golden Ranking” concept**, which methodically ranks patch tokens basedon scoring functions. A key innovation in our method isthe **deployment of a lightweight predictor**, trained to closely approximate this Golden Ranking. Furthermore, to mitigatethe inevitable performance loss resulting from the pruningprocess, **we integrate learnable text and visual tokens intoour framework.** These tokens, especially visual tokens, wereshown to play a pivotal role in compensating for potential performance degradation, ensuring the model’s output remainsaccurate post-pruning. Our extensive experiments across avariety of datasets have demonstrated that our framework canachieve **a substantial reduction in patch tokens, by up to 40%in CLIP’s ViT**, while maintaining comparable performance(only 0.3% lower accuracy)."*

**2.4.1 Phase 1 - Identifying the pruning subset**
Inside the CLIP ViT, not all the patches identify for relevant information, so the idea is to find a **token subset** that is useless for the model through a ranking system formed by 3 different scores.

*"To bridge this gap, we introduce three scoring metrics designed to rank patch tokens based on their impact on CLIP’s predictions, forming what we term the ‘Golden Ranking‘."*

To do this, they apply a series of pruned tokens to the model and evaluate:

1. **Label-Driven Ranking Score** *"The pruned tokens T i are scored based on CLIP’s zero-shot posterior probabil- ity of assigning the pruned sequence X T ¯ i to the ground- 33.2. Phase II: Predicting the Golden Ranking truth label $y_{gt}$"*
$$
s(\mathcal{T_i}) = \mathcal{P(\mathcal{y_gt}|\mathcal{X_{\mathcal{T^{¯}_i}}})}
$$
2. **Maximum Confidence Score** *"assess the pruned tokens T i based on the maximum confidence across all classes $y$."*
$$
s(\mathcal{T_i}) = \max_{y}\mathcal{P(y|X_{\mathcal{T^{¯}_i}})}
$$
3. **Feature Preservation Score** *"feature preservation seeks to identify the tokens that, when re- moved, do not alter the image representation, as ex- pressed by the CLS token embedding. This score is quantified using cosine similarity"*
$$
s(\mathcal{T_i}) = \frac{\mathcal{Z^{cls}}\mathcal{Z^{cls}_{\mathcal{T^{¯}_i}}}} {||\mathcal{Z^{cls}}|| ||\mathcal{Z^{cls}_{\mathcal{T^{¯}_i}}}||}
$$

Where Z cls denotes the CLS embedding obtained from the full sequence $\mathcal{X}$ and $\mathcal{Z^{cls}_{\mathcal{T^{¯}_i}}}$ denotes the embedding obtained with a pruned sequence.

*"Instead, we remove a larger r × r block of tokens, resulting in more noticeable changes. As the removal block T i slides over the image, each token is removed and assessed multiple times, thus stabilizing the final average score of each token t"*

$$
\mathcal{s(t)} = \frac{1}{|\mathcal{T_i}|} \sum_{i:t \epsilon \mathcal{T_i}} \mathcal{s(\mathcal{T_i})} 
$$

**2.4.2 Phase 2**
*"After establishing the Golden Ranking using one of the three metrics above, we train a lightweight predictor, $ŝ = h(Z; Θ) ∈ ℜ^N$ , to efficiently approximate it, and thus identify the least useful tokens from their representations Z."*

They now train a predictor to insert in the early layers of the ViT to prune a given number of tokens, this predictor tries to emulate the Golden Ranking scores

**2.4.3 Phase 3**
Prompt tuning (aka need some training) to compensate the pruning done in Phase2.
The prompt tuning is made by adding learnable prompts both at the beginning of the Visual encoder and Text encoder, where they are correlated by a linear transformation described by:
$\mathcal{P_v^i = MP_t^i}$

---

---

- Reinforce Data
	- Come hanno fatto il dataset reinforcement? Con che logica?

- fanno training `multi-model reinforced training`
	- cosa significa?
	- cross-modal afifnity mimicking \[68] cosa significa?
	- **extend uni-modal model ensembling** to multimodal setup [33, 46]
	- **we extend the dataset reinforcement strategy** to the multi-modal setup [14 ]
- loss function
- synthetic captions
	- CoCa [74] -> due nozioni su questo
- significato di strong teacher in un ensemble teacher

- architettura di mobile clip
	- structura reparametrizatoin in image and text encoders -> spiegare bene cosa hanno fatto 
	- Text-RepMixer (guarda 2.2.1 nel nostro MobileClip)
		- come funziona
		- cosa sono i filtering networks [16]?
		- e' basato su RepMixer
			- come funziona?
		- migliorie rispetto RepMixer
		-  **hybrid text encoder(Conv/Transf) which makes use of 1-D convolutions and self-attention layers**: _Text-RepMixer_ which decouples train-time and inference-time architectures.
			- Decouple train-time and inference-time -> HOW?
		- implementation
		- Appendice E
	- MCi
		- based on FastViT [62]
		- guarda il 2.2.2
		- guarda appendice A


- PuMer (token pruning and merging)
- TinyClip
	- vedere la loss e come si potrebbe integrare