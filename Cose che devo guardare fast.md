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
	3. inference
		1. prumer (towards model efficiency)
			1. rendere il modello piu' veloce (forse occupa meno vram? da vedere)
	4. architecture (?)
		1. Sigmoid Self-Attention -> NotebookLM salvaci tu

## 2 Possibili Migliorie

### 2.1 Dataset Reinforcement

### 2.2 TinyCLIP - Loss

*TinyClip contributions*

**2.2.1 Affinity Mimicking**
This is the technique (introduced by a loss) that mixed the embeddings from the teacher and the student model, using the classic contrastive loss between both the Text2Image and viceversa among the teacher and the student

$$
L_{distill} = L_{I2T} + L_{T2I}\\
L_{I2T} = CE(A^{s}_{I2T}, A^{t}_{I2T})\\
A_{I2T}(i,j) = \frac{exp(I_{i} * T_{j}/ \tau)} {\sum_{k \epsilon \Beta}exp(I_{i} * T_{k}\tau)}
$$

**2.2.2 Weight Inheritance - Manual and Automatic**
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

**2.2.3 Progressive Multi-Stage Distillation**
*"When attempting to achieve a high target sparsity, i.e.,>70%, compressing the model in a single stage can lead toa significant reduction in accuracy and even result in con-vergence failure. This is due to the fact that most weights ofthe large model are directly discarded, including those thatare important for ensuring model quality and convergence.As a solution, we propose a multi-stage progressive distil-lation method to achieve a high compression rate withoutseriously sacrificing accuracy. In each stage, we use a mod-est degree of compression, e.g., 25%, to avoid large loss ofperformance and make training stable."*

Just using the two precedent methods gradually, maybe changing also the percentage of compression. 

### 2.2 Loss integration

We know that std. CLIP loss is:

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

So two plausible Losses might be:

$$
\mathcal{L}_{\text{Total}}(\mathcal{B}) 
= (1 - \lambda)\,\mathcal{L}_{\text{CLIP}}(\mathcal{B})
  + \lambda((1 - \alpha)\mathcal{L}_{\text{Distill}}(\mathcal{B}) + \alpha\mathcal{L}_{\text{Sparsity}}(\mathcal{B}))   
$$

where:
- $\lambda$ balanced as usual the std. CLIP loss and the distillation one
- $\alpha$ instead balance the contribution of distill and sparsity
- $\mathcal{L}_{\text{Distill}}$ could be the MobileClip or the TinyClip loss, in both cases using the stored embeddings
- $\mathcal{L}_{\text{Sparsity}}$ empowers the compression of the transformer architecture, eventually applied only to the self-attention layers in case of the hybrid transformer encoder and to the ConvFFN 

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