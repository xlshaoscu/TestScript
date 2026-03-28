"""
查看模型 tokenizer 信息的脚本
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from transformers import AutoTokenizer, AutoConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def inspect_tokenizer(model_path: str):
    """检查 tokenizer 的详细信息"""
    print(f"\n{'='*60}")
    print(f"检查模型: {model_path}")
    print(f"{'='*60}\n")

    # 1. 加载模型配置
    print("--- 模型配置 ---")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"vocab_size (from config): {getattr(config, 'vocab_size', 'N/A')}")
        print(f"model_type: {getattr(config, 'model_type', 'N/A')}")
        print(f"architectures: {getattr(config, 'architectures', 'N/A')}")
    except Exception as e:
        print(f"加载配置失败: {e}")

    # 2. 加载 fast tokenizer
    print("\n--- Fast Tokenizer ---")
    try:
        tokenizer_fast = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        print(f"类型: {type(tokenizer_fast)}")
        print(f"is_fast: {tokenizer_fast.is_fast}")
        print(f"vocab_size: {tokenizer_fast.vocab_size}")
        print(f"len(tokenizer): {len(tokenizer_fast)}")
        
        vocab = tokenizer_fast.get_vocab()
        max_token_id = max(vocab.values()) if vocab else 0
        print(f"max_token_id (from vocab): {max_token_id}")
        print(f"vocab 数量: {len(vocab)}")
        
        # 特殊 token
        print(f"bos_token: {tokenizer_fast.bos_token} (id: {tokenizer_fast.bos_token_id})")
        print(f"eos_token: {tokenizer_fast.eos_token} (id: {tokenizer_fast.eos_token_id})")
        print(f"pad_token: {tokenizer_fast.pad_token} (id: {tokenizer_fast.pad_token_id})")
        print(f"unk_token: {tokenizer_fast.unk_token} (id: {tokenizer_fast.unk_token_id})")
        
        # 添加的 token
        added_tokens = tokenizer_fast.get_added_vocab()
        if added_tokens:
            print(f"\n添加的 token 数量: {len(added_tokens)}")
            print(f"添加的 token: {list(added_tokens.keys())[:10]}...")  # 只显示前10个
    except Exception as e:
        print(f"加载 fast tokenizer 失败: {e}")

    # 3. 加载 slow tokenizer
    print("\n--- Slow Tokenizer ---")
    try:
        tokenizer_slow = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"类型: {type(tokenizer_slow)}")
        print(f"is_fast: {tokenizer_slow.is_fast}")
        print(f"vocab_size: {tokenizer_slow.vocab_size}")
        print(f"len(tokenizer): {len(tokenizer_slow)}")
        
        vocab = tokenizer_slow.get_vocab()
        max_token_id = max(vocab.values()) if vocab else 0
        print(f"max_token_id (from vocab): {max_token_id}")
    except Exception as e:
        print(f"加载 slow tokenizer 失败: {e}")

    # 4. 检查 tokenizer 文件
    print("\n--- Tokenizer 文件 ---")
    from pathlib import Path
    model_dir = Path(model_path)
    
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.tok.json",
    ]
    
    for f in tokenizer_files:
        file_path = model_dir / f
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  {f}: 存在 ({size:,} bytes)")
        else:
            print(f"  {f}: 不存在")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # 示例：检查本地模型
    model_path = "/opt/data/models/IDEA-Research/Rex-Omni"
    
    # 如果模型路径不存在，可以换成其他模型测试
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请修改 model_path 为你的模型路径")
        # 可以使用 HuggingFace 模型 ID
        # model_path = "Qwen/Qwen2.5-7B-Instruct"
    
    inspect_tokenizer(model_path)
