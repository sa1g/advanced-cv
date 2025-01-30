![[reinforce data scheem.png]]
- strategy to improve a dataset once such that the accuracy of any model architecture trained on the reinforced dataset is improved at no additional training cost for users
- based on data augmentation and knowledge distillation

It needs to satify:
- No overhead for users: Minimal increase in the computational cost of training a new model for similar total iterations
- Minimal changes in user code and mo
- Architecture independence: Improve the test accuracy across variety of model architectures


## Data Augmentation
- standard Inception-style augmentation: random resized crop and random horizontal flipping
- incorporate mixing augmentations; MixUp, CutMix 
- automatic augmentation methods: RandAugment, AutoAugment; to generate new data

Negative side:
- fails to satify all the desiderata as it does not provide architecture independent generalization

## Knowledge distillation
- training of a student model by matching the output of a teacher model
- has shown to improve the accuracy of new models independent of their architecture significantly more than data augmentations 

Negative side:
- expensive: requires performing the infer ence (forward-pass) of an often significantly large teacher model at every training iteration
- requires modifying the training code to perform two forward passes on both the teacher and the student
	- fails to satisfy minimal overhead and code change desiderata

## Dataset Reinforcement
We precompute and store the output of a strong pretrained model on multiple augmentations per sample as reinforcements. The stored outputs are more informative and useful for training compared with ground truth labels.

goal is to find generalizable reinforcements that improve the accuracy of any architecture

The reinforced dataset consists of the original dataset plus the reinforcement meta data for all training samples. During the reinforcement process, for each sample a fixed number of reinforcements is generated using parametrized augmen- tation operations and evaluating the teacher predictions. To save storage, instead of storing the augmented images, the augmentation parameters are stored alongside the sparsified output of the teacher. As a result, the extra storage needed is only a fraction of the original training set for large datasets

One of the benefits of dataset reinforcement paradigm is that the teacher can be expensive to train and use as long as we can afford to run it once on the target dataset for reinforce- ment. Also, the process of dataset reinforcement is highly parallelizable because performing the forward-pass on the teacher to generate predictions on multiple augmentations does not depend on any state or any optimization trajectory

we suggest the following guidelines: 1) use ensemble of strong teachers trained on large diverse data 2) balance reinforcement difficulty and model complexity

ab. 7 shows that models trained using ImageNet+ dataset are up to 20% more robust. Overall, these robustness results in conjunction with results in Tab. 4 highlight the effective- ness of the proposed dataset

## Conclusion
unwraps tradeoffs in find- ing generalizable reinforcements controlled by the difficulty of augmentations and we propose ways to balancs

demonstrate significant improvements in robustness, calibration and transfer.
Our novel method of training and fine-tuning on doubly re- inforced datasets (e.g., ImageNet + to CIFAR-100+ ) demon- strates new possibilities of DR as a generic strategy

Our desiderata would also be satisfied by methods that expand the training data, especially in limited data domains, using strong generative foundation models

### Limitations
Limitations of the teacher can potentially transfer through dataset reinforcement. For example, over- confident biased teachers should not be used and diverse ensembles are preferred. Human verification of the reinforce- ments is also a solution. Note that original labels are unmod- ified in reinforced datasets and can be used in curriculums. Our robustness and transfer learning evaluations consistently show better transfer and generalization for ImageNet+ mod- els likely because of lower bias of the teacher ensemble trained on diverse data.