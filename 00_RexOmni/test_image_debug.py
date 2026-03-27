from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import logging
from msprobe.pytorch import PrecisionDebugger, seed_all
import os
from rex_omni.parser import parse_prediction
from rex_omni import RexOmniVisualize


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


# 1. 检查tokenizer和模型配置
model_path = "/opt/data/models/IDEA-Research/Rex-Omni"
print("=" * 60)
print("Checking tokenizer and model configuration...")
print("=" * 60)

tokenizer = None
config = None
processor = None

try:
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 加载processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=16 * 28 * 28,
        max_pixels=256 * 28 * 28,
        trust_remote_code=True
    )

    # 检查词汇表
    vocab = tokenizer.get_vocab()
    max_token_id = max(vocab.values()) if vocab else 0
    vocab_size = len(vocab) if vocab else 0

    print(f"[INFO] Tokenizer vocab size: {vocab_size}")
    print(f"[INFO] Tokenizer max token ID: {max_token_id}")
    print(f"[INFO] Model config vocab size: {getattr(config, 'vocab_size', 'Not found')}")

    # 检查特殊token
    special_tokens = tokenizer.special_tokens_map
    print(f"[INFO] Special tokens: {list(special_tokens.keys())}")

    # 检查是否有超出范围的token
    if max_token_id > 151936:
        print(f"[WARNING] Tokenizer has tokens beyond model vocab (151936)")
        problematic = [(t, i) for t, i in vocab.items() if i >= 151936]
        print(f"[WARNING] {len(problematic)} tokens have ID >= 151936")
        for token, token_id in problematic[:10]:  # 只显示前10个
            print(f"  ID {token_id}: '{repr(token)}'")

    # 检查processor和tokenizer是否一致
    if hasattr(processor, 'tokenizer'):
        proc_vocab = processor.tokenizer.get_vocab()
        proc_max_id = max(proc_vocab.values()) if proc_vocab else 0
        print(f"[INFO] Processor tokenizer max ID: {proc_max_id}")

except Exception as e:
    logger.error(f"Error checking tokenizer/config: {e}")
    import traceback
    traceback.print_exc()

# 2. 初始化 vLLM 模型（添加调试参数）
print("\n" + "=" * 60)
print("Initializing LLM with debugging parameters...")
print("=" * 60)

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

# 3. 设置采样参数
sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
    skip_special_tokens=False,
)

# 4. 准备图像
image_path = "/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg"
print(f"\n[INFO] Loading image from: {image_path}")
image = Image.open(image_path).convert("RGB")
print(f"[INFO] Image size: {image.size}, mode: {image.mode}")

# 5. 构建多模态消息
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
                "min_pixels": 16 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            },
            {"type": "text", "text": "Detect person, car. Output the bounding box coordinates in [x0, y0, x1, y1] format."}
        ]
    }
]

# 6. 处理输入（详细调试）
print("\n" + "=" * 60)
print("Processing multimodal input...")
print("=" * 60)

try:
    # 6.1 使用processor构建文本
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"[INFO] Generated text length: {len(text)} chars")
    print(f"[INFO] First 200 chars: {text[:200]}...")

    # 检查文本中的特殊token
    import re
    special_pattern = r'<[^>]+>|\[[^\]]+\]'
    found_special = re.findall(special_pattern, text)
    if found_special:
        print(f"[INFO] Found special tokens in text: {list(set(found_special))}")

    # 6.2 处理图像
    image_inputs, video_inputs = process_vision_info(messages)
    print(f"[INFO] Image inputs type: {type(image_inputs)}")
    if hasattr(image_inputs, 'shape'):
        print(f"[INFO] Image inputs shape: {image_inputs.shape}")
    if hasattr(image_inputs, 'dtype'):
        print(f"[INFO] Image inputs dtype: {image_inputs.dtype}")

    # 6.3 构建 vLLM 输入
    llm_inputs = {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs}
    }

    print(f"[INFO] LLM inputs keys: {list(llm_inputs.keys())}")

except Exception as e:
    logger.error(f"Error processing multimodal input: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 7. 生成（详细错误处理）
print("\n" + "=" * 60)
print("Starting generation...")
print("=" * 60)

try:
    # 开启数据dump（可选）
    # model_exec = model.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
    # debugger.start(model=model_exec)

    print("[INFO] Calling model.generate()...")
    outputs = model.generate([llm_inputs], sampling_params=sampling_params)
    print("[SUCCESS] Generation completed!")

except Exception as e:
    logger.exception(f"[ERROR] Generation failed: {e}")
    print("\n" + "=" * 60)
    print("Debugging information:")
    print("=" * 60)

    # 提供调试建议
    print("\n1. Check if the error is related to tokenizer vocabulary:")
    print("   - Try setting vocab_size to actual model vocab size")
    print("   - Check if processor adds special tokens beyond model vocab")

    print("\n2. Check if the error is related to multimodal input:")
    print("   - Try simple text-only input first")
    print("   - Check image processing pipeline")

    print("\n3. Check model compatibility:")
    print("   - Ensure model supports vLLM")
    print("   - Check if model requires special initialization")

    # 关闭数据dump（如果开启）
    # debugger.stop()
    # debugger.step()

    exit(1)

# 8. 获取结果
print("\n" + "=" * 60)
print("Generation Results:")
print("=" * 60)

try:
    generated_text = outputs[0].outputs[0].text
    print(f"Generated text length: {len(generated_text)} chars")
    print("\n" + "-" * 40)
    print("Generated text:")
    print("-" * 40)
    print(generated_text)

    # 检查输出中是否有特殊格式
    import re
    bbox_pattern = r'\[.*?\]'
    bboxes = re.findall(bbox_pattern, generated_text)
    if bboxes:
        print(f"\nFound {len(bboxes)} bounding box(es):")
        for i, bbox in enumerate(bboxes):
            print(f"  {i+1}. {bbox}")
    # 8.1 解析预测结果
    print("\n" + "=" * 60)
    print("Parsing Predictions:")
    print("=" * 60)
    w, h = image.size
    extracted_predictions = parse_prediction(
        text=generated_text,
        w=w,
        h=h,
        task_type="detection"
    )
    print(f"Image size: {w}x{h}")
    print(f"Parsed predictions: {extracted_predictions}")

    # 8.2 可视化结果
    print("\n" + "=" * 60)
    print("Visualizing Results:")
    print("=" * 60)
    vis_image = RexOmniVisualize(
        image=image,
        predictions=extracted_predictions,
        font_size=20,
        draw_width=5,
        show_labels=True,
    )
    vis_output_path = os.path.join(os.path.dirname(__file__), "visualize_output.jpg")
    vis_image.save(vis_output_path)
    print(f"[INFO] Visualization saved to: {vis_output_path}")        

except Exception as e:
    logger.error(f"Error processing results: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)