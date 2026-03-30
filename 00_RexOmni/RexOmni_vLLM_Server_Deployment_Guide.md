# Rex-Omni vLLM Server 部署问题解决指南

## 问题描述

在使用 vLLM server 部署 Rex-Omni 模型时，通过 API 调用返回的响应内容为 `"person,,,,,"` 占位符，而不是实际的坐标值。然而，使用本地脚本 `validate_model.py` 却能正常工作并返回正确的坐标。

**错误响应示例：**
```json
{
  "choices": [{
    "message": {
      "content": "person,,,,,",
      "role": "assistant"
    }
  }]
}
```

**期望的响应格式：**
```text
<|object_ref_start|>person<|object_ref_end|><|box_start|><123><456><789><234><|box_end|>
```

## 根本原因分析

通过对比 `validate_model.py`（成功）和 vLLM server API（失败）的差异，发现以下关键问题：

### 1. vLLM Server 启动参数格式错误

**错误的命令：**
```bash
--limit-mm-per-prompt image=1
```

**问题分析：**
`--limit-mm-per-prompt` 参数需要 JSON 格式字符串，但使用了 `image=1` 这种 shell 格式，导致 vLLM 无法正确解析多模态配置。

### 2. API 请求格式不匹配

**错误的请求格式（OpenAI 兼容格式）：**
```python
{
  "type": "image_url",
  "image_url": {"url": "data:image/jpeg;base64,..."}
}
```

**问题分析：**
Rex-Omni（基于 Qwen2.5-VL）期望的是 Qwen-VL 专用格式，而不是标准的 OpenAI 格式。

### 3. 缺少必要的图像预处理参数

**缺失的参数：**
- `min_pixels`：最小像素数（16×28×28 = 12544）
- `max_pixels`：最大像素数（256×28×28 = 200704）

**问题分析：**
没有指定图像尺寸限制，导致图像可能没有被正确缩放和预处理。

### 4. 生成参数不完整

**缺失的生成控制参数：**
- `top_p`：核采样参数（应为 0.05）
- `top_k`：Top-k 采样（应为 1）
- `repetition_penalty`：重复惩罚（应为 1.05）

**问题分析：**
只设置了 `temperature=0`，但模型需要完整的生成参数才能产生正确的坐标输出。

## 解决方案概览

1. **修正 vLLM server 启动命令** - 使用正确的 JSON 格式和多模态参数
2. **使用正确的 API 请求格式** - 采用 Qwen-VL 专用格式
3. **添加完整的图像预处理参数** - 指定像素范围
4. **设置完整的生成参数** - 与本地运行保持一致

## 正确的 vLLM Server 启动命令

```bash
vllm serve /opt/data/models/IDEA-Research/Rex-Omni \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --tokenizer-mode slow \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 10}' \  # 关键：JSON 格式
    --min-pixels 12544 \                     # 16 × 28 × 28
    --max-pixels 200704                      # 256 × 28 × 28
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--limit-mm-per-prompt` | `'{"image": 10}'` | **必须使用 JSON 格式**，启用多模态支持，每提示最多10张图像 |
| `--min-pixels` | `12544` | 最小像素数，与模型训练一致 |
| `--max-pixels` | `200704` | 最大像素数，防止图像过大 |
| `--trust-remote-code` | 无值 | 必需，用于加载 Qwen-VL 自定义代码 |
| `--tokenizer-mode` | `slow` | 使用完整 tokenizer，支持特殊标记 |

### 验证服务器启动成功

启动后检查日志，确认有以下信息：
```
Using multimodal processor for Qwen2.5-VL
limit_mm_per_prompt set to: {'image': 10}
Loading model weights...
Model loaded successfully
```

## 正确的 API 调用示例

### 方案1：使用 Qwen-VL 专用格式（推荐）

```python
import base64
import requests
import json
from PIL import Image
import io

def test_vllm_server_qwen_format():
    """使用 Qwen-VL 专用格式调用 vLLM server"""

    # 1. 加载并准备图像
    image_path = "/path/to/your/image.jpg"

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size  # 保存原始尺寸用于坐标转换

        # 转换为 base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 2. 构建 Qwen-VL 格式消息
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_base64,  # 纯 base64，无 data:image/jpeg;base64, 前缀
                    "min_pixels": 12544,   # 16 × 28 × 28
                    "max_pixels": 200704   # 256 × 28 × 28
                },
                {
                    "type": "text",
                    "text": "Detect all persons. Output the bounding box coordinates in [x0, y0, x1, y1] format."
                }
            ]
        }
    ]

    # 3. 发送请求
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": "/opt/data/models/IDEA-Research/Rex-Omni",
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
            "max_tokens": 2048,
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']

        # 4. 解析坐标（模型输出为 bin 格式 [0-999]）
        import re
        coord_pattern = r'<(\d+)><(\d+)><(\d+)><(\d+)>'
        matches = re.findall(coord_pattern, content)

        print(f"原始输出: {content[:200]}...")

        if matches:
            print(f"\n找到 {len(matches)} 个边界框:")
            for i, match in enumerate(matches, 1):
                x0, y0, x1, y1 = map(int, match)
                # 转换为绝对坐标
                x0_abs = (x0 / 999.0) * w
                y0_abs = (y0 / 999.0) * h
                x1_abs = (x1 / 999.0) * w
                y1_abs = (y1 / 999.0) * h
                print(f"  框{i}: [{x0_abs:.1f}, {y0_abs:.1f}, {x1_abs:.1f}, {y1_abs:.1f}]")
        else:
            print("\n警告：未找到标准坐标格式")

        return result
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    test_vllm_server_qwen_format()
```

### 方案2：使用 OpenAI 兼容格式（备选）

如果 Qwen-VL 格式不可用，尝试以下 OpenAI 兼容格式：

```python
import base64
import requests

def test_vllm_server_openai_format():
    """使用 OpenAI 兼容格式调用 vLLM server"""

    image_path = "/path/to/your/image.jpg"

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": "/opt/data/models/IDEA-Research/Rex-Omni",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "low"  # 添加 detail 参数
                            }
                        },
                        {
                            "type": "text",
                            "text": "Detect all persons. Output the bounding box coordinates in [x0, y0, x1, y1] format."
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
            "max_tokens": 2048,
        },
        timeout=60
    )

    print(response.json())
    return response.json()
```

## 完整的端到端解决方案

### 步骤1：停止并重新启动 vLLM server

```bash
# 1. 查找并停止当前运行的 vLLM server
pkill -f "vllm serve"

# 2. 使用正确参数重新启动
vllm serve /opt/data/models/IDEA-Research/Rex-Omni \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --tokenizer-mode slow \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 10}' \
    --min-pixels 12544 \
    --max-pixels 200704

# 3. 确认启动成功（观察日志）
# 应该看到以下信息：
# - "Using multimodal processor for Qwen2.5-VL"
# - "limit_mm_per_prompt set to: {'image': 10}"
# - "Model loaded successfully"
```

### 步骤2：运行测试脚本

创建 `test_rexomni_vllm.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rex-Omni vLLM Server 测试脚本
验证服务器部署和 API 调用功能
"""

import base64
import requests
import json
import time
import re
from PIL import Image
import io
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class RexOmniVLLMClient:
    """Rex-Omni vLLM Server 客户端"""

    def __init__(self, base_url="http://127.0.0.1:8000", model_path="/opt/data/models/IDEA-Research/Rex-Omni"):
        self.base_url = base_url.rstrip('/')
        self.model_path = model_path
        self.api_url = f"{self.base_url}/v1/chat/completions"

    def prepare_image(self, image_path):
        """准备图像：转换为 base64 并获取尺寸"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            # 转换为 base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            logger.info(f"图像加载成功: {image_path}, 尺寸: {w}x{h}")
            return img_base64, w, h

    def call_qwen_format(self, image_path, categories="person", task="detection"):
        """使用 Qwen-VL 格式调用 API"""

        img_base64, w, h = self.prepare_image(image_path)

        # 构建提示词
        if task == "detection":
            prompt = f"Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
        elif task == "pointing":
            prompt = f"Point to {categories}."
        else:
            prompt = f"Perform {task} for {categories}."

        # Qwen-VL 格式消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_base64,
                        "min_pixels": 12544,
                        "max_pixels": 200704
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 发送请求
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
            "max_tokens": 2048,
        }

        logger.info(f"发送请求到: {self.api_url}")
        logger.info(f"任务: {task}, 类别: {categories}")

        start_time = time.time()
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            elapsed = time.time() - start_time
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            return None

        if response.status_code != 200:
            logger.error(f"API 错误: {response.status_code}")
            logger.error(f"响应: {response.text}")
            return None

        result = response.json()
        content = result['choices'][0]['message']['content']
        usage = result.get('usage', {})

        logger.info(f"请求完成，耗时: {elapsed:.2f}s")
        logger.info(f"生成 token 数: {usage.get('completion_tokens', 'N/A')}")
        logger.info(f"原始输出前200字符: {content[:200]}...")

        # 解析结果
        parsed_result = self.parse_result(content, w, h, task)

        return {
            "success": True,
            "raw_output": content,
            "parsed_predictions": parsed_result,
            "image_size": (w, h),
            "task": task,
            "categories": categories,
            "response_time": elapsed,
            "token_usage": usage
        }

    def parse_result(self, content, w, h, task):
        """解析模型输出"""

        if task in ["detection", "pointing", "visual_prompting"]:
            # 查找标准格式：<|object_ref_start|>category<|object_ref_end|><|box_start|>coords<|box_end|>
            pattern = r'<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>'
            matches = re.findall(pattern, content, re.DOTALL)

            result = {}
            for category, coords_text in matches:
                category = category.strip()

                # 提取坐标数字
                coord_pattern = r'<(\d+)>'
                coord_matches = re.findall(coord_pattern, coords_text)

                annotations = []
                coord_strings = coords_text.split(',')

                for coord_str in coord_strings:
                    coord_nums = re.findall(coord_pattern, coord_str.strip())

                    if len(coord_nums) == 4:  # 边界框
                        x0, y0, x1, y1 = map(int, coord_nums)
                        x0_abs = (x0 / 999.0) * w
                        y0_abs = (y0 / 999.0) * h
                        x1_abs = (x1 / 999.0) * w
                        y1_abs = (y1 / 999.0) * h
                        annotations.append({
                            "type": "box",
                            "coords": [x0_abs, y0_abs, x1_abs, y1_abs]
                        })
                    elif len(coord_nums) == 2:  # 点
                        x, y = map(int, coord_nums)
                        x_abs = (x / 999.0) * w
                        y_abs = (y / 999.0) * h
                        annotations.append({
                            "type": "point",
                            "coords": [x_abs, y_abs]
                        })

                if annotations:
                    result[category] = annotations

            return result

        return {"raw_content": content}

    def test_connection(self):
        """测试服务器连接"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("服务器连接正常")
                return True
        except:
            pass

        try:
            # 尝试 vLLM 的健康检查端点
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM API 连接正常")
                return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")

        return False

def main():
    parser = argparse.ArgumentParser(description="测试 Rex-Omni vLLM Server")
    parser.add_argument("--image", required=True, help="测试图像路径")
    parser.add_argument("--categories", default="person", help="检测类别")
    parser.add_argument("--task", default="detection", choices=["detection", "pointing"], help="任务类型")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="vLLM server URL")

    args = parser.parse_args()

    client = RexOmniVLLMClient(base_url=args.url)

    # 测试连接
    if not client.test_connection():
        logger.error("无法连接到 vLLM server，请确认服务器正在运行")
        return

    # 执行推理
    result = client.call_qwen_format(
        image_path=args.image,
        categories=args.categories,
        task=args.task
    )

    if result and result["success"]:
        logger.info("=" * 60)
        logger.info("推理成功！")
        logger.info(f"图像尺寸: {result['image_size']}")
        logger.info(f"任务: {result['task']}")
        logger.info(f"类别: {result['categories']}")
        logger.info(f"响应时间: {result['response_time']:.2f}s")

        if result["parsed_predictions"]:
            logger.info("\n解析结果:")
            for category, annotations in result["parsed_predictions"].items():
                logger.info(f"  {category}:")
                for ann in annotations:
                    if ann["type"] == "box":
                        coords = ann["coords"]
                        logger.info(f"    边界框: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]")
                    elif ann["type"] == "point":
                        coords = ann["coords"]
                        logger.info(f"    点: [{coords[0]:.1f}, {coords[1]:.1f}]")
        else:
            logger.warning("未解析到有效坐标，原始输出:")
            logger.warning(result["raw_output"][:500])
    else:
        logger.error("推理失败")

if __name__ == "__main__":
    main()
```

### 步骤3：运行测试

```bash
# 1. 确保 vLLM server 正在运行
# 2. 运行测试脚本
python test_rexomni_vllm.py \
    --image /home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg \
    --categories "person" \
    --task detection \
    --url http://127.0.0.1:8000
```

## 故障排除指南

### 常见问题及解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 返回 `"person,,,,,"` | 1. `--limit-mm-per-prompt` 格式错误<br>2. API 格式不匹配<br>3. 缺少图像参数 | 1. 使用 `'{"image": 10}'` 格式<br>2. 使用 Qwen-VL 格式<br>3. 添加 `min_pixels` 和 `max_pixels` |
| 服务器启动失败 | 模型路径错误或权限问题 | 检查模型路径，确保 vLLM 有读取权限 |
| 内存不足 | GPU 内存不足 | 减小 `--gpu-memory-utilization` (如 0.7) |
| 响应时间过长 | 图像太大或模型未优化 | 确保使用 `--enforce-eager`，检查图像尺寸 |
| 无法解析坐标 | 输出格式不符合预期 | 检查提示词是否匹配任务类型 |

### 调试步骤

1. **检查服务器日志**
   ```bash
   # 查看实时日志
   tail -f /var/log/vllm/server.log
   ```

2. **验证多模态支持**
   ```bash
   # 检查进程参数
   ps aux | grep vllm | grep -v grep
   # 应该看到 --limit-mm-per-prompt '{"image": 10}'
   ```

3. **测试简单请求**
   ```python
   # 纯文本测试，验证基础功能
   import requests
   response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
       "model": "/opt/data/models/IDEA-Research/Rex-Omni",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 10
   })
   print(response.json())
   ```

4. **检查图像预处理**
   ```python
   # 验证图像 base64 编码
   import base64
   with open("test.jpg", "rb") as f:
       data = base64.b64encode(f.read()).decode()
   print(f"Base64 长度: {len(data)}")
   ```

## 对比分析：本地运行 vs vLLM Server

| 维度 | 本地运行 (`validate_model.py`) | vLLM Server API |
|------|-------------------------------|-----------------|
| **初始化** | 使用 `AutoProcessor.from_pretrained()` | 依赖 vLLM server 配置 |
| **图像处理** | `process_vision_info()` + `smart_resize()` | 通过 API 参数传递 |
| **消息格式** | Qwen-VL 专用格式 | 需要显式指定格式 |
| **坐标系统** | bin 坐标 [0-999] 转绝对坐标 | 相同，需要客户端转换 |
| **性能** | 单次推理，适合调试 | 批量推理，适合生产 |

## 最佳实践

1. **始终使用正确的参数格式**
   - `--limit-mm-per-prompt '{"image": 10}'`（JSON 格式）
   - `--min-pixels 12544` 和 `--max-pixels 200704`

2. **统一使用 Qwen-VL 格式**
   - `type: "image"` 而不是 `type: "image_url"`
   - 纯 base64，无 `data:image/jpeg;base64,` 前缀

3. **保持生成参数一致**
   - `temperature: 0.0`
   - `top_p: 0.05`
   - `top_k: 1`
   - `repetition_penalty: 1.05`

4. **实现客户端解析逻辑**
   - 模型输出为 bin 坐标 [0-999]
   - 需要转换为绝对像素坐标
   - 使用正则表达式提取坐标

5. **监控和日志**
   - 记录请求/响应时间
   - 捕获解析错误
   - 监控 token 使用量

## 总结

Rex-Omni 模型在 vLLM server 部署中返回 `"person,,,,,"` 的问题主要源于：

1. **启动参数格式错误**：`--limit-mm-per-prompt` 需要使用 JSON 格式
2. **API 格式不匹配**：需要使用 Qwen-VL 专用格式而非 OpenAI 格式
3. **参数不完整**：缺少图像预处理和完整的生成参数

通过修正启动命令、使用正确的 API 格式和添加必要的参数，可以确保 vLLM server 正确部署 Rex-Omni 模型，并返回准确的坐标输出。

## 附录

### 相关文件路径
- **模型路径**: `/opt/data/models/IDEA-Research/Rex-Omni`
- **测试图像**: `/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg`
- **本地验证脚本**: `01_WorkDir/TestScript/00_RexOmni/validate_model.py`

### 参考文档
- [Rex-Omni 官方 GitHub](https://github.com/IDEA-Research/Rex-Omni)
- [Qwen2.5-VL 文档](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [vLLM 多模态支持](https://docs.vllm.ai/en/latest/serving/multi_modal.html)

### 更新记录
| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-03-30 | 1.0 | 初始版本，解决 vLLM server 部署问题 |
| 2026-03-30 | 1.1 | 添加完整测试脚本和故障排除指南 |