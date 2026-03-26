from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
from msprobe.pytorch import PrecisionDebugger, seed_all



# 在模型训练开始前固定随机性
seed_all()
# 请勿将PrecisionDebugger的初始化流程插入到循环代码中
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")


# 1. 初始化 vLLM 模型
model_path = "/opt/data/models/IDEA-Research/Rex-Omni"

model = LLM(
    model=model_path,
    tokenizer=model_path,
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=1,
    enforce_eager=True,
    dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=16 * 28 * 28,
    max_pixels=256 * 28 * 28,
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

# 3. 准备图像
image = Image.open("/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg").convert("RGB")

#image = Image.open("test.jpg").convert("RGB")

# 4. 构建多模态消息
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

# 5. 处理输入
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

# 6. 构建 vLLM 输入
llm_inputs = {
    "prompt": text,
    "multi_modal_data": {"image": image_inputs}
}

# 7. 生成
print("*************************************")
try:
    # 开启数据dump, 指定采集推理模型逐字符循环推理中的第1~3次
    #debugger.start(model=model, token_range=[1, 100])
    outputs = model.generate([llm_inputs], sampling_params=sampling_params)
except Exception as e:
    # 关闭数据dump并落盘
    debugger.stop()
    debugger.step()


# 8. 获取结果
generated_text = outputs[0].outputs[0].text
print(generated_text)
