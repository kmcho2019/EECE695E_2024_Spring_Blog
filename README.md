# EECE695E_2024_Spring_Blog: VeRA: Vector-based Random Matrix Adaptation
EECE695E_2024_Spring_Blog for Efficient Machine Learning Class w/ Sejin Park and Kyumin Cho

## Introduction to LoRA (Low Rank Adaptation) family of PEFT (Parameter Efficient Finetuning)
Large language Models or LLMs consists of at least billions of parameters. This makes it extremely expensive to train and finetune. For example, the weights of GPT-3 175B can take up to 350GB when stored in FP16 precision (2 bytes per FP16 x 175B=350GB). When training such models additional information such as the optimizer states and gradients are needed. Assuming FP32 training with AdamW optimizer a single weight parameter of the model, it requires 4 bytes to store the weight itself in FP32, 8 bytes per parameter to for the optimizer AdamW (where two states, first moment and second moment in FP32, are maintained for each parameter), and 4 bytes per parameter to store the gradient in FP32. This adds up to 16 bytes of storage space needed for each model parameter required for training. (Source: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#anatomy-of-models-memory). This means that a full finetune of a small model such as Llama-3 8B can take 128GB (16 bytes x 8B = 128GB) just to store the parameters. This calculation excludes the forward activations as well as the training batch data. This makes even training relatively small models impossible on a single GPU as datacenter-class GPUs such as A100 or H100 max out at 80GB for GPU and especially for consumber level GPUs such as RTX 4090 which only has 24GB.

Not only does the weights needs to be stored on the GPU VRAM during VRAM, each finetune version of the model needs to store the entire modified copy. This means even if mass-storage devices like HDDs are used, it becomes prohibitively impossible to store multiple custom finetune version of the data itself.

Therefore parameter efficient finetuning or PEFT methods have been developed that is able to finetune and specialize LLMs by only using small amount of parameters, this not only reduces the number of GPUs required to train the model itself, it cuts down on the permanent storage capacity required to store multiple versions of it. The most popular of these approaches is low rank adaptation or LoRA. As its name suggests, this technique uses low-rank tensors to represent large matrices in LLMs. The hidden dimension size in LLMs gets very large with size with GPT-3 175B having a hidden dimension (d) of 12,288. By multiplying two matrices with extremely low rank (as low as 1 or 2), it is possible to represent changes to the large matrix. 

This can be shown in the following diagram.
(Insert Diagram of LoRA here)


Since only the differences to the original model are tracked in training, original model parameters can be frozen and only the small low-rank tensors need to be trained. Gradients or optimizer states don't are not required for the original model, only for the small low-rank tensors, so this greatly reduces the GPU VRAM requirement. Also, when servicing large variations of custom finetune models, only a single copy of the large model needs to be stored and each version only needs to store the small low-rank tensors that represents the difference between the original weights. This makes servicing large number of variations feasible without astronomical cost for each customer.

## How VeRA works
Even with parameter efficient nature of LoRA it still requires a non-trivial amount of storage for each version. If a custom version was wanted for each vendor or consumer the storage requirement can easily add up. VeRA tries to take advanatage of random initialization of basis to reduce the number of unique parameters needed for each finetune.

## Performance 

## Extensions

### DVoRA (DoRA + VeRA)

## Future Avenue of Research

### NAS to VeRA

### Better initialization  

### Future universal random weights 
Platonic model weights hyptohesis. If large fundamental models share a common representation, it is possible that there could be an ideal way to represent the randomized matrix basis on which VeRA operates well in.
https://arxiv.org/abs/2405.07987 
