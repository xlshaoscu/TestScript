"""
验证 slow tokenizer 是否能正确处理 Rex-Omni 模型
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


def verify_tokenizer_compatibility(model_path: str):
    """验证 tokenizer 与模型的兼容性"""
    print(f"\n{'='*70}")
    print(f"验证 tokenizer 兼容性: {model_path}")
    print(f"{'='*70}\n")

    # 1. 获取模型 vocab_size
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_vocab_size = config.vocab_size
    print(f"模型 vocab_size: {model_vocab_size}")

    # 2. 测试 Fast tokenizer
    print("\n--- Fast Tokenizer 测试 ---")
    try:
        tok_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        vocab_fast = tok_fast.get_vocab()
        max_id_fast = max(vocab_fast.values())
        
        print(f"max_token_id: {max_id_fast}")
        print(f"是否超出模型范围: {'是 ❌' if max_id_fast >= model_vocab_size else '否 ✓'}")
        print(f"超出数量: {max_id_fast - model_vocab_size + 1 if max_id_fast >= model_vocab_size else 0}")
        
        # 检查哪些 token ID 超出范围
        out_of_range_tokens = [(k, v) for k, v in vocab_fast.items() if v >= model_vocab_size]
        if out_of_range_tokens:
            print(f"\n超出范围的 token (前20个):")
            for token, tid in out_of_range_tokens[:20]:
                print(f"  {token}: {tid}")
    except Exception as e:
        print(f"Fast tokenizer 错误: {e}")

    # 3. 测试 Slow tokenizer
    print("\n--- Slow Tokenizer 测试 ---")
    try:
        tok_slow = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        vocab_slow = tok_slow.get_vocab()
        max_id_slow = max(vocab_slow.values())
        
        print(f"max_token_id: {max_id_slow}")
        print(f"是否超出模型范围: {'是 ❌' if max_id_slow >= model_vocab_size else '否 ✓'}")
        print(f"超出数量: {max_id_slow - model_vocab_size + 1 if max_id_slow >= model_vocab_size else 0}")
        
        # 检查哪些 token ID 超出范围
        out_of_range_tokens = [(k, v) for k, v in vocab_slow.items() if v >= model_vocab_size]
        if out_of_range_tokens:
            print(f"\n超出范围的 token (前20个):")
            for token, tid in out_of_range_tokens[:20]:
                print(f"  {token}: {tid}")
    except Exception as e:
        print(f"Slow tokenizer 错误: {e}")

    # 4. 测试编码解码
    print("\n--- 编码解码测试 ---")
    test_texts = [
        "Hello, world!",
        "你好，世界！",
        "<|im_start|>user\nHello<|im_end|>",
    ]
    
    for text in test_texts:
        print(f"\n原文: {text}")
        
        # Fast
        try:
            ids_fast = tok_fast.encode(text, add_special_tokens=False)
            decoded_fast = tok_fast.decode(ids_fast)
            max_id = max(ids_fast) if ids_fast else 0
            in_range = max_id < model_vocab_size
            print(f"  Fast: ids={ids_fast[:10]}..., max_id={max_id}, in_range={in_range}")
        except Exception as e:
            print(f"  Fast 错误: {e}")
        
        # Slow
        try:
            ids_slow = tok_slow.encode(text, add_special_tokens=False)
            decoded_slow = tok_slow.decode(ids_slow)
            max_id = max(ids_slow) if ids_slow else 0
            in_range = max_id < model_vocab_size
            print(f"  Slow: ids={ids_slow[:10]}..., max_id={max_id}, in_range={in_range}")
        except Exception as e:
            print(f"  Slow 错误: {e}")

    # 5. 结论
    print("\n" + "="*70)
    print("结论:")
    if max_id_fast >= model_vocab_size and max_id_slow < model_vocab_size:
        print("✓ 建议使用 tokenizer_mode='slow'")
        print("  原因: Fast tokenizer 的 max_token_id 超出模型 vocab_size")
    elif max_id_fast < model_vocab_size:
        print("✓ 可以使用 Fast tokenizer")
    else:
        print("⚠ 两个 tokenizer 都有问题，请检查模型配置")
    print("="*70 + "\n")


if __name__ == "__main__":
    model_path = "/opt/data/models/IDEA-Research/Rex-Omni"
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请修改 model_path 为你的模型路径")
    else:
        verify_tokenizer_compatibility(model_path)
