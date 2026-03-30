#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex-Omni vLLM 服务器测试（客户端预处理版）
直接使用 validate_model.py 的预处理流程，通过 vLLM 服务器 API 进行推理

原理：
1. 在客户端使用 AutoProcessor 和 process_vision_info 进行图像预处理（与 validate_model.py 完全相同）
2. 将预处理后的 image_embeds 发送到 vLLM 服务器
3. 服务器需启用 --enable-mm-embeds 标志

使用方法：
    1. 启动 vLLM 服务器并启用图像嵌入支持：
       vllm serve /opt/data/models/IDEA-Research/Rex-Omni \
           --max-model-len 4096 \
           --gpu-memory-utilization 0.8 \
           --dtype float16 \
           --tokenizer-mode slow \
           --trust-remote-code \
           --limit-mm-per-prompt '{"image": 10}' \
           --enable-mm-embeds \
           --host 0.0.0.0 \
           --port 8000

    2. 运行本脚本：
       python vllm_server_test_with_preprocessing.py
"""

import os
import io
import base64
import logging
import pickle
from typing import Dict, Any, Tuple

import torch
import requests
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 配置参数
BASE_URL = "http://localhost:8000"
MODEL_PATH = "/opt/data/models/IDEA-Research/Rex-Omni"
IMAGE_PATH = "/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg"


class RexOmniClientPreprocessor:
    """Rex-Omni 客户端预处理器，复制 validate_model.py 的预处理逻辑"""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.8,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.processor = None

    def initialize(self) -> bool:
        """初始化处理器，与 validate_model.py 完全一致"""
        logger.info(f"初始化处理器，模型路径: {self.model_path}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=16 * 28 * 28,
                max_pixels=256 * 28 * 28,
                trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"
            logger.info("处理器加载成功")
            return True
        except Exception as e:
            logger.error(f"处理器初始化失败: {e}")
            return False

    def preprocess_like_validate_model(
        self,
        image_path: str,
        task: str = "detection",
        categories: str = "person"
    ) -> Dict[str, Any]:
        """
        完全复制 validate_model.py 的预处理流程

        返回:
            Dict 包含:
            - prompt: 处理后的文本提示
            - image_embeds: 预处理后的图像张量（base64 编码）
            - image_size: 原始图像尺寸
            - image_inputs: 原始图像输入（用于调试）
        """
        logger.info(f"加载图像: {image_path}")

        # 1. 加载图像（与 validate_model.py 第99行相同）
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        logger.info(f"图像尺寸: {w}x{h}")

        # 2. 构建提示词（与 validate_model.py 第103行相同）
        prompt = f"Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format."

        # 3. 构建消息（与 validate_model.py 第105-119行相同）
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
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 4. 应用聊天模板（与 validate_model.py 第121-126行相同）
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"提示词生成完成，长度: {len(text)} 字符")

        # 5. 处理视觉信息（与 validate_model.py 第128行相同）
        image_inputs, _ = process_vision_info(messages)

        # 记录预处理后的图像形状
        if hasattr(image_inputs, 'shape'):
            logger.info(f"图像预处理完成，形状: {image_inputs.shape}")
        elif isinstance(image_inputs, dict):
            logger.info(f"图像预处理完成，类型: 字典，键: {list(image_inputs.keys())}")
        else:
            logger.info(f"图像预处理完成，类型: {type(image_inputs)}")

        # 6. 将图像张量编码为 base64
        image_embeds_base64 = self._encode_image_inputs(image_inputs)

        return {
            "image_size": (w, h),
            "prompt": prompt,
            "processed_text": text,
            "image_embeds": image_embeds_base64,
            "image_inputs": image_inputs,  # 保留原始输入用于调试
            "task": task,
            "categories": categories
        }

    def _encode_image_inputs(self, image_inputs) -> str:
        """
        将图像输入编码为 base64 字符串

        支持多种输入类型:
        - torch.Tensor: 使用 torch.save 序列化
        - dict: 使用 pickle 序列化
        - 其他: 尝试 pickle 序列化
        """
        try:
            if isinstance(image_inputs, torch.Tensor):
                # 张量序列化
                buffer = io.BytesIO()
                torch.save(image_inputs, buffer)
                buffer.seek(0)
                data = buffer.read()
            elif isinstance(image_inputs, dict):
                # 字典序列化
                data = pickle.dumps(image_inputs)
            else:
                # 其他类型序列化
                data = pickle.dumps(image_inputs)

            return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            logger.error(f"图像输入编码失败: {e}")
            # 返回空字符串表示失败
            return ""

    def create_server_request(
        self,
        preprocessed_data: Dict[str, Any],
        use_processed_text: bool = False
    ) -> Dict[str, Any]:
        """
        创建发送到 vLLM 服务器的请求

        参数:
            use_processed_text: 是否使用处理器生成的文本（包含聊天模板）
                              如果为 False，则使用原始提示词
        """
        # 选择文本内容
        if use_processed_text:
            text_content = preprocessed_data["processed_text"]
            # 注意：使用 processed_text 时，消息结构需要调整
            # 因为 processed_text 已经包含了聊天模板
            messages = [{"role": "user", "content": text_content}]
        else:
            text_content = preprocessed_data["prompt"]
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": preprocessed_data["image_embeds"],
                    },
                    {"type": "text", "text": text_content}
                ]
            }]

        # 构建请求
        request = {
            "model": MODEL_PATH,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
            "skip_special_tokens": False,
        }

        return request


def check_server_health() -> bool:
    """检查 vLLM 服务器健康状态"""
    logger.info("检查服务器健康状态...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            logger.info("服务器健康状态正常")
            return True
        else:
            logger.warning(f"服务器返回非200状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"服务器健康检查失败: {e}")
        return False


def send_to_vllm_server(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """发送请求到 vLLM 服务器"""
    logger.info("发送请求到 vLLM 服务器...")

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
            timeout=120  # 较长的超时时间，因为图像嵌入可能较大
        )

        logger.info(f"服务器响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            return {"success": True, "data": result}
        else:
            logger.error(f"服务器返回错误: {response.status_code}")
            logger.error(f"响应内容: {response.text[:500]}...")
            return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}

    except requests.exceptions.Timeout:
        logger.error("请求超时")
        return {"success": False, "error": "请求超时"}
    except requests.exceptions.ConnectionError:
        logger.error("连接服务器失败")
        return {"success": False, "error": "连接失败"}
    except Exception as e:
        logger.error(f"请求异常: {e}")
        return {"success": False, "error": str(e)}


def parse_detection_results(result: Dict[str, Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    解析检测结果

    参数:
        result: 服务器返回的 JSON 结果
        image_size: 图像原始尺寸 (width, height)

    返回:
        解析后的结果字典
    """
    if "choices" not in result or len(result["choices"]) == 0:
        return {"error": "无效的响应格式"}

    content = result["choices"][0]["message"]["content"]
    w, h = image_size

    logger.info(f"原始输出内容: {content}")

    # 尝试解析边界框
    # 预期格式: [x0, y0, x1, y1] 每行一个
    bboxes = []
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            try:
                # 移除方括号并分割
                coords = line[1:-1].split(',')
                if len(coords) == 4:
                    x0, y0, x1, y1 = map(float, coords)
                    bboxes.append({
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "normalized": x1 <= 1.0 and y1 <= 1.0  # 检查是否归一化坐标
                    })
            except ValueError as e:
                logger.warning(f"解析坐标失败: {line}, 错误: {e}")

    return {
        "raw_content": content,
        "bboxes": bboxes,
        "bbox_count": len(bboxes),
        "image_size": image_size,
        "lines_count": len(lines)
    }


def compare_with_original_preprocessing():
    """
    与原始预处理方法对比

    此函数展示两种方法的不同：
    1. 使用 image_embeds（客户端预处理）
    2. 使用 image_url（服务器端预处理）
    """
    logger.info("=" * 60)
    logger.info("与原始方法对比测试")
    logger.info("=" * 60)

    # 方法1：客户端预处理（image_embeds）
    logger.info("方法1: 客户端预处理 (image_embeds)")
    preprocessor = RexOmniClientPreprocessor(MODEL_PATH)

    if not preprocessor.initialize():
        logger.error("预处理器初始化失败")
        return

    preprocessed = preprocessor.preprocess_like_validate_model(IMAGE_PATH)
    request_data = preprocessor.create_server_request(preprocessed, use_processed_text=False)

    result1 = send_to_vllm_server(request_data)

    if result1["success"]:
        parsed1 = parse_detection_results(result1["data"], preprocessed["image_size"])
        logger.info(f"方法1结果: {parsed1['bbox_count']} 个边界框")
        for i, bbox in enumerate(parsed1["bboxes"]):
            logger.info(f"  框 {i+1}: [{bbox['x0']:.2f}, {bbox['y0']:.2f}, {bbox['x1']:.2f}, {bbox['y1']:.2f}]")
    else:
        logger.error(f"方法1失败: {result1['error']}")

    logger.info("-" * 40)

    # 方法2：服务器端预处理（image_url，原始方法）
    logger.info("方法2: 服务器端预处理 (image_url)")
    with open(IMAGE_PATH, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    request_data2 = {
        "model": MODEL_PATH,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": "Detect all persons. Output only [x0,y0,x1,y1] per line."}
            ]
        }],
        "temperature": 0,
        "max_tokens": 40000
    }

    result2 = send_to_vllm_server(request_data2)

    if result2["success"]:
        # 获取图像尺寸用于解析
        image = Image.open(IMAGE_PATH).convert("RGB")
        w, h = image.size

        parsed2 = parse_detection_results(result2["data"], (w, h))
        logger.info(f"方法2结果: {parsed2['bbox_count']} 个边界框")
        for i, bbox in enumerate(parsed2["bboxes"]):
            logger.info(f"  框 {i+1}: [{bbox['x0']:.2f}, {bbox['y0']:.2f}, {bbox['x1']:.2f}, {bbox['y1']:.2f}]")
    else:
        logger.error(f"方法2失败: {result2['error']}")

    logger.info("=" * 60)
    logger.info("对比完成")

    # 总结对比
    if result1["success"] and result2["success"]:
        count1 = parsed1["bbox_count"]
        count2 = parsed2["bbox_count"]
        logger.info(f"客户端预处理边界框数量: {count1}")
        logger.info(f"服务器端预处理边界框数量: {count2}")

        if count1 > 0 and count2 == 0:
            logger.info("✅ 客户端预处理方法成功检测到边界框，而服务器端方法失败")
        elif count1 == 0 and count2 > 0:
            logger.info("❌ 服务器端预处理方法成功检测到边界框，而客户端方法失败")
        elif count1 > 0 and count2 > 0:
            logger.info("✅ 两种方法都检测到了边界框")
        else:
            logger.info("❌ 两种方法都未能检测到边界框")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Rex-Omni vLLM 服务器测试（客户端预处理版）")
    logger.info("=" * 60)

    # 检查服务器状态
    if not check_server_health():
        logger.error("vLLM 服务器未运行或不可达")
        logger.error("请确保服务器已启动并启用 --enable-mm-embeds 标志")
        logger.error("启动命令示例：")
        logger.error("vllm serve /opt/data/models/IDEA-Research/Rex-Omni \\")
        logger.error("    --max-model-len 4096 \\")
        logger.error("    --gpu-memory-utilization 0.8 \\")
        logger.error("    --dtype float16 \\")
        logger.error("    --tokenizer-mode slow \\")
        logger.error("    --trust-remote-code \\")
        logger.error("    --limit-mm-per-prompt '{\"image\": 10}' \\")
        logger.error("    --enable-mm-embeds \\")
        logger.error("    --host 0.0.0.0 \\")
        logger.error("    --port 8000")
        return

    # 运行对比测试
    compare_with_original_preprocessing()

    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()