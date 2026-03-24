#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLLM API 测试脚本 - 针对多模态模型 Rex-Omni

说明：
1. 假设 VLLM 服务器已启动，端口 8091
2. 需要安装 requests: pip install requests
3. 图像需要转换为 base64 编码
"""

import base64
import json
import logging
import requests
from PIL import Image
from io import BytesIO

# 配置 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def image_to_base64(image_path, format="JPEG"):
    """将图像转换为 base64 字符串"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_vllm_api(image_path, categories, task="detection"):
    """测试 VLLM API 调用"""

    # VLLM 服务器地址
    base_url = "http://127.0.0.1:8091"

    # 1. 转换图像为 base64
    image_base64 = image_to_base64(image_path)

    # 2. 构建消息（参考 rex_test.py 格式）
    # Rex-Omni 是多模态模型，需要特殊格式
    
    # 方法2：尝试 VLLM 原生格式（如果 OpenAI 格式不支持多模态）
    logger.info("方法2：测试 /generate (VLLM 原生格式)")

    # 构建类似 rex_test.py 中的 llm_inputs
    # 注意：VLLM 可能需要特殊的 multi_modal_data 格式
    prompt_text = f"Detect {', '.join(categories)}. Output the bounding box coordinates in [x0, y0, x1, y1] format."

    # 尝试 VLLM 的多模态输入格式
    # 从 rex_test.py 看，需要 multi_modal_data: {"Image": image_inputs}
    # 但 HTTP API 可能需要 base64 编码的图像数组

    payload_vllm = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": [image_base64]  # 假设 VLLM 接受 base64 图像数组
        },
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 0.05,
        "top_k": 1,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "stop": ["<|im_end|>"]
    }

    # 尝试不同的端点
    endpoints = [
        f"{base_url}/generate",  # VLLM 原生端点
        f"{base_url}/v1/generate",  # 可能的其他端点
        f"{base_url}/v1/completions",  # 补全端点
    ]

    for endpoint in endpoints:
        logger.info(f"尝试端点: {endpoint}")
        try:
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload_vllm),
                timeout=30
            )
            logger.info(f"状态码: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"成功！响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                logger.error(f"失败！响应: {response.text}")
        except Exception as e:
            logger.error(f"端点 {endpoint} 调用异常: {e}")

    # 方法3：尝试通过消息模板构建（Qwen-VL 格式）
    logger.info("方法3：测试 Qwen-VL 格式消息")

    # 构建 Qwen-VL 格式的消息
    messages_qwen = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_base64,
                    "min_pixels": 16 * 28 * 28,  # 12544
                    "max_pixels": 2560 * 28 * 28,  # 2007040
                },
                {
                    "type": "text",
                    "text": f"Detect {', '.join(categories)}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
                }
            ]
        }
    ]

    payload_qwen = {
        "model": "/opt/data/models/IDEA-Research/Rex-Omni",
        "messages": messages_qwen,
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 0.05,
        "top_k": 1,
        "repetition_penalty": 1.05,
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload_qwen, ensure_ascii=False),
            timeout=30
        )
        logger.info(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"成功！响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            logger.error(f"失败！响应: {response.text}")
    except Exception as e:
        logger.error(f"Qwen 格式调用异常: {e}")

    logger.error("所有方法都失败！请检查：")
    logger.error("1. VLLM 服务器是否运行: curl http://127.0.0.1:8091/v1/models")
    logger.error("2. 模型名称是否正确")
    logger.error("3. VLLM 是否支持多模态 API")
    logger.error("4. 查看 VLLM 服务器日志获取详细错误")

    return None

def check_server_status():
    """检查 VLLM 服务器状态"""
    base_url = "http://127.0.0.1:8091"

    logger.info("检查 VLLM 服务器状态...")

    # 检查 /v1/models 端点
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            logger.info(f"服务器正常，可用模型: {models}")
            return True
        else:
            logger.error(f"服务器响应异常: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"无法连接到服务器: {e}")
        return False

    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            logger.info("Swagger 文档可用")
    except:
        pass

    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=10)
        if response.status_code == 200:
            logger.info("OpenAPI 规范可用")
    except:
        pass

def main():
    """主函数"""

    if not check_server_status():
        logger.error("请先启动 VLLM 服务器:")
        logger.error("python -m vllm.entrypoints.openai.api_server \\")
        logger.error("  --model IDEA-Research/Rex-Omni \\")
        logger.error("  --served-model-name Rex-Omni \\")
        logger.error("  --limit-mm-per-prompt '{\"image\": 10}' \\")
        logger.error("  --max-model-len 4096")
        return

    image_path = "/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg"
    categories = ["person", "car", "dog"]
    task = "detection"

    logger.info("开始测试 API 调用...")
    logger.info(f"图像: {image_path}")
    logger.info(f"任务: {task}")
    logger.info(f"类别: {categories}")

    result = test_vllm_api(image_path, categories, task)

    if result:
        logger.info("测试成功！")
        if "choices" in result:
            for choice in result["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    logger.info(f"模型输出: {content}")
        elif "text" in result:
            logger.info(f"模型输出: {result['text']}")
    else:
        logger.error("测试失败！")

    logger.info("附加调试命令:")
    logger.info("1. 查看可用模型: curl http://127.0.0.1:8091/v1/models")
    logger.info("2. 查看 API 文档: curl http://127.0.0.1:8091/docs")
    logger.info("3. 查看 OpenAPI 规范: curl http://127.0.0.1:8091/openapi.json")
    logger.info("4. 测试纯文本请求:")
    logger.info('   curl http://127.0.0.1:8091/v1/chat/completions \\')
    logger.info('     -H "Content-Type: application/json" \\')
    logger.info('     -d \'{"model": "Rex-Omni", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}\'')

if __name__ == "__main__":
    main()
