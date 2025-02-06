# Build your nanoGPT！

## 🛠️Installation

Clone this repo:

```bash
git clone https://github.com/xrlexpert/Build-your-nanoGPT
```

Build up the environment

```bash
cd ./Build-your-nanoGPT
conda create -n nanogpt python==3.8
conda activate nanogpt
```

Install `Pytorch>=2.0` following the [official guide](https://pytorch.org/get-started/previous-versions/)

Then install the dependencies in requirements.txt

```bash
pip install requirements.txt
```

## 📊 Data Preparation

This project in alignment with [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt), uses the sample dataset from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) as the pre-training dataset.

You can directly run the following script to download it, which additionally performs pre-tokenization on the text.

```bash
python fine_web.py
```

## 📖 Code Tutorial

For a better understanding of the original project and this implementation, refer to the following resources:

- 🎥 **Video Explanation of the Original Project**: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
  A detailed video walkthrough of the original project's concepts and implementation. 
- 📑 **In-depth Blog on Code Explanation**: [Build nanoGPT - Hirox's blog](https://xrlexpert.github.io/2025/02/05/Build-nanoGPT/)
  A blog post providing a detailed breakdown of the code, including its structure, logic, and modifications in this project. 

**Both are highly recommended for a deeper dive into the theory and implementation.**

The training process is configured via `config.yaml`. You can modify hyperparameters such as batch size, learning rate, and training epochs in this file.

🚀To train the model on a single node, run the following command:

```bash
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=8
    train_ddp.py 
```

Once the model is trained, you can use it to generate text with the following command:

```python
python inference.py --path "/path/to/your/model_checkpoint.pt" --input "Your input text here"
```

The model will generate the next token using the **top-5 sampling** method, where it randomly selects from the top 5 tokens based on their probabilities.

I have provided the [pre-trained model weights](https://mega.nz/file/TLYGQISL#T9Hs7RDQ_zicVx7mQ8BsfiBBfY7z6m1zSKWWCv7jM48) after training on **2 A100  80G GPUs**. You can download and use these weights for inference. Simply specify the path to the model checkpoint in the `--path` argument when running the inference script.

## 🪧 Acknowledgement

This project is built upon the work done in the following repositories:

- [**build-nanogpt**](https://github.com/karpathy/build-nanogpt) by Karpathy, which provided invaluable insights into building and training smaller-scale GPT models.
- [**LLMs-from-scratch**](https://github.com/rasbt/LLMs-from-scratch) by Rasbt, for its comprehensive guide and implementation of training large language models from scratch.

 Thanks for their great work!

## 📇 License

MIT
