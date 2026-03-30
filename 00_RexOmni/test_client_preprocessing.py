#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex-Omni Client-Side Preprocessing Test
Preprocess image on client side using process_vision_info(),
then send to standard vLLM server via image_embeds

Usage:
    1. Start vLLM server with --enable-mm-embeds flag:
       vllm serve /opt/data/models/IDEA-Research/Rex-Omni \\
           --max-model-len 4096 \\
           --gpu-memory-utilization 0.8 \\
           --dtype float16 \\
           --tokenizer-mode slow \\
           --trust-remote-code \\
           --limit-mm-per-prompt '{"image": 10}' \\
           --enable-mm-embeds \\
           --host 0.0.0.0 \\
           --port 8000

    2. Run this test script:
       python test_client_preprocessing.py
"""

import os
import io
import base64
import logging
from typing import Dict, Any

import torch
import requests
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
MODEL_PATH = "/opt/data/models/IDEA-Research/Rex-Omni"
IMAGE_PATH = "/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg"


class ClientSidePreprocessor:
    """Preprocess image on client side before sending to vLLM server"""

    def __init__(
        self,
        model_path: str,
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
    ):
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = None

    def initialize(self) -> bool:
        """Initialize processor for preprocessing"""
        logger.info(f"Initializing processor from: {self.model_path}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                trust_remote_code=True
            )
            logger.info("Processor loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            return False

    def preprocess_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Preprocess image using the same pipeline as validate_model.py
        
        Returns:
            Dict containing:
            - text: processed prompt with chat template
            - image_embeds: preprocessed image tensor (base64 encoded)
            - image_grid_thw: image grid dimensions
        """
        w, h = image.size
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"Prompt generated, length: {len(text)} chars")

        image_inputs, _ = process_vision_info(messages)
        
        logger.info(f"Image preprocessed, shape: {image_inputs.shape if hasattr(image_inputs, 'shape') else 'N/A'}")

        image_embeds_base64 = self._tensor_to_base64(image_inputs)
        
        return {
            "text": text,
            "image_embeds": image_embeds_base64,
            "image_grid_thw": self._get_grid_thw(image_inputs),
            "original_size": (w, h)
        }

    def _tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """Convert tensor to base64 string"""
        import pickle
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _get_grid_thw(self, tensor: torch.Tensor) -> list:
        """Get grid THW from tensor shape"""
        if hasattr(tensor, 'shape'):
            if len(tensor.shape) == 4:
                return [1, tensor.shape[1], tensor.shape[2]]
            elif len(tensor.shape) == 3:
                return [1, tensor.shape[0], tensor.shape[1]]
        return [1, 1, 1]


def test_health():
    """Test health endpoint"""
    logger.info("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        logger.info(f"Health status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def test_with_client_preprocessing(
    image_path: str,
    categories: str = "person",
    task: str = "detection"
) -> Dict[str, Any]:
    """
    Test vLLM server with client-side preprocessing
    
    This approach:
    1. Preprocesses image on client side using process_vision_info()
    2. Sends preprocessed image_embeds to vLLM server
    """
    logger.info("=" * 60)
    logger.info("Testing with client-side preprocessing")
    logger.info("=" * 60)

    preprocessor = ClientSidePreprocessor(
        model_path=MODEL_PATH,
        min_pixels=16 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    if not preprocessor.initialize():
        logger.error("Failed to initialize preprocessor")
        return {}

    logger.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    logger.info(f"Image size: {w}x{h}")

    prompt = f"Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format."

    logger.info("Preprocessing image on client side...")
    preprocessed = preprocessor.preprocess_image(image, prompt)

    logger.info("Sending request to vLLM server...")
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL_PATH,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_embeds",
                            "image_embeds": preprocessed["image_embeds"],
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
        },
        timeout=120
    )

    logger.info(f"Response status: {response.status_code}")
    result = response.json()

    if "choices" in result:
        content = result["choices"][0]["message"]["content"]
        logger.info(f"Generated content: {content}")
    else:
        logger.warning(f"Unexpected response: {result}")

    return {
        "status_code": response.status_code,
        "response": result,
        "original_size": (w, h)
    }


def test_without_preprocessing(
    image_path: str,
    categories: str = "person",
) -> Dict[str, Any]:
    """
    Test vLLM server without client-side preprocessing (standard approach)
    
    This sends raw base64 image directly to server
    """
    logger.info("=" * 60)
    logger.info("Testing without client-side preprocessing (standard)")
    logger.info("=" * 60)

    logger.info(f"Loading image: {image_path}")
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format."

    logger.info("Sending request to vLLM server...")
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL_PATH,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
        },
        timeout=120
    )

    logger.info(f"Response status: {response.status_code}")
    result = response.json()

    if "choices" in result:
        content = result["choices"][0]["message"]["content"]
        logger.info(f"Generated content: {content}")
    else:
        logger.warning(f"Unexpected response: {result}")

    return {
        "status_code": response.status_code,
        "response": result
    }


def main():
    """Main test entry point"""
    logger.info("=" * 60)
    logger.info("Rex-Omni Client-Side Preprocessing Test")
    logger.info("=" * 60)

    if not test_health():
        logger.error("vLLM server is not running. Please start it first:")
        logger.error("  vllm serve /opt/data/models/IDEA-Research/Rex-Omni ...")
        return

    logger.info("-" * 60)
    result_with_preprocessing = test_with_client_preprocessing(IMAGE_PATH)

    logger.info("-" * 60)
    result_without_preprocessing = test_without_preprocessing(IMAGE_PATH)

    logger.info("=" * 60)
    logger.info("Comparison Results:")
    logger.info("=" * 60)
    
    if "choices" in result_with_preprocessing.get("response", {}):
        content_with = result_with_preprocessing["response"]["choices"][0]["message"]["content"]
        logger.info(f"With preprocessing: {content_with}")
    
    if "choices" in result_without_preprocessing.get("response", {}):
        content_without = result_without_preprocessing["response"]["choices"][0]["message"]["content"]
        logger.info(f"Without preprocessing: {content_without}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
