from vllm import LLM, SamplingParams
import torch
import logging
from msprobe.pytorch import PrecisionDebugger, seed_all
import os
from transformers import AutoTokenizer, AutoConfig


os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 配置 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 在模型训练开始前固定随机性
seed_all()
# 请勿将PrecisionDebugger的初始化流程插入到循环代码中
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")


# 1. 初始化 vLLM 模型
model_path = "/opt/data/models/IDEA-Research/Rex-Omni"

# 先检查tokenizer和模型配置
print("Checking tokenizer and model configuration...")
tokenizer = None
config = None
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False  # 尝试禁用fast tokenizer
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    vocab = tokenizer.get_vocab()
    max_token_id = max(vocab.values()) if vocab else 0
    vocab_size = len(vocab) if vocab else 0

    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Tokenizer max token ID: {max_token_id}")
    print(f"Model config vocab size: {getattr(config, 'vocab_size', 'Not found')}")

    # 检查是否有特殊token导致ID过大
    special_tokens = tokenizer.special_tokens_map
    print(f"Special tokens: {list(special_tokens.keys())}")

    # 打印前10个和后10个token的ID
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    print("\nFirst 10 tokens:")
    for token, idx in sorted_items[:10]:
        print(f"  {idx}: {repr(token)}")

    print("\nLast 10 tokens:")
    for token, idx in sorted_items[-10:]:
        print(f"  {idx}: {repr(token)}")

except Exception as e:
    print(f"Error checking tokenizer/config: {e}")

# 尝试不同的初始化方式
print("\nInitializing LLM with explicit vocab_size...")
model = LLM(
    model=model_path,
    tokenizer=model_path,
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=1,
    enforce_eager=True,
    dtype=torch.float16,
    tokenizer_mode="slow",  # 使用slow tokenizer
)

# 2. 设置采样参数
sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
    skip_special_tokens=False,
)

# 3. 构建纯文本消息
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {
        "role": "user",
        "content": "Hello, tell me something about yourself."
    }
]

# 4. 使用tokenizer处理纯文本输入
# 对于多模态模型，我们使用tokenizer而不是processor来处理纯文本
try:
    # 使用之前创建的tokenizer，如果没有则重新创建
    if 'tokenizer' not in locals():
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

    # 使用tokenizer应用chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Using tokenizer for text processing")
except Exception as e:
    print(f"Error using tokenizer: {e}")
    # 回退到简单的方式：手动构建输入
    text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n<|im_start|>user\n{messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"

# 6. 构建 vLLM 输入（纯文本，没有多模态数据）
llm_inputs = {
    "prompt": text,
    # 不传递multi_modal_data
}

# 7. 生成
print("*************************************")
print(f"Input text: {text[:100]}...")
try:
    # 开启数据dump, 指定采集推理模型逐字符循环推理中的第1~3次
    model_exec = model.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
    #debugger.start(model=model_exec)
    outputs = model.generate([llm_inputs], sampling_params=sampling_params)
except Exception as e:
    logger.exception(f" 生成文本时发生异常: {e}")
    # 关闭数据dump并落盘
    #debugger.stop()
    #debugger.step()


# 8. 获取结果
generated_text = outputs[0].outputs[0].text
print("Generated text:")
print(generated_text)