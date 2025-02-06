import torch
import argparse
import tiktoken
from model import GPTModel, generate_and_print_sample

# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Load and run GPT model for text generation.")
    parser.add_argument('--path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input text to generate from')
    return parser.parse_args()

# 主程序
def main():
    # 解析命令行参数
    args = parse_args()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-key-value bias
    }

    # 加载模型
    model = GPTModel(GPT_CONFIG_124M)
    checkpoint = torch.load(args.path)  # 使用命令行传入的路径
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["config"])

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 使用GPT-2的tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 生成文本并输出
    generate_and_print_sample(model, tokenizer, device, GPT_CONFIG_124M["context_length"], args.input)

if __name__ == "__main__":
    main()
