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
		1. implementare tinyvit -> obiettivo di ridurre ancora piu' il modello -> less latency, less training time
	3. inference
		1. prumer (towards model efficiency)
			1. rendere il modello piu' veloce (forse occupa meno vram? da vedere)
	4. architecture (?)
		1. Sigmoid Self-Attention -> NotebookLM salvaci tu



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