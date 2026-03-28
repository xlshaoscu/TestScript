#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex-Omni Model Validation Script
Focus on verifying model correctness with vLLM backend
"""

import os
import logging
from PIL import Image
from typing import Dict, Any

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from rex_omni.parser import parse_prediction
from rex_omni import RexOmniVisualize

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RexOmniValidator:
    """Validator for Rex-Omni model"""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.8,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model = None
        self.processor = None
        self.sampling_params = None

    def initialize(self) -> bool:
        """Initialize model and processor"""
        logger.info(f"Initializing model from: {self.model_path}")
        
        try:
            self.model = LLM(
                model=self.model_path,
                tokenizer=self.model_path,
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=1,
                enforce_eager=True,
                dtype=torch.float16,
                tokenizer_mode="slow",
            )
            logger.info("vLLM model loaded successfully")

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=16 * 28 * 28,
                max_pixels=256 * 28 * 28,
                trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"
            logger.info("Processor loaded successfully")

            self.sampling_params = SamplingParams(
                max_tokens=2048,
                temperature=0.0,
                top_p=0.05,
                top_k=1,
                repetition_penalty=1.05,
                skip_special_tokens=False,
            )
            
            return True

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False

    def run_inference(
        self,
        image_path: str,
        task: str = "detection",
        categories: str = "person"
    ) -> Dict[str, Any]:
        """Run inference on a single image"""
        logger.info(f"Loading image: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        logger.info(f"Image size: {w}x{h}")

        prompt = f"Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
        
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

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"Prompt generated, length: {len(text)} chars")

        image_inputs, _ = process_vision_info(messages)
        
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs}
        }

        logger.info("Starting inference...")
        outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
        
        generated_text = outputs[0].outputs[0].text
        logger.info(f"Generation completed, output length: {len(generated_text)} chars")

        extracted_predictions = parse_prediction(
            text=generated_text,
            w=w,
            h=h,
            task_type=task
        )
        logger.info(f"Parsed predictions: {extracted_predictions}")

        return {
            "image_size": (w, h),
            "prompt": prompt,
            "raw_output": generated_text,
            "predictions": extracted_predictions,
            "image": image
        }

    def visualize_results(
        self,
        result: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Visualize and save detection results"""
        try:
            vis = RexOmniVisualize(
                image=result["image"],
                predictions=result["predictions"],
                font_size=20,
                draw_width=5,
                show_labels=True,
            )
            vis.save(output_path)
            logger.info(f"Visualization saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to visualize: {e}")
            return False


def main():
    """Main validation entry point"""
    model_path = "/opt/data/models/IDEA-Research/Rex-Omni"
    image_path = "/home/s00964975/00_Software/Rex-Omni/tutorials/detection_example/test_images/boys.jpg"
    output_path = os.path.join(os.path.dirname(__file__), "validation_output.jpg")

    logger.info("Starting Rex-Omni model validation")

    validator = RexOmniValidator(model_path=model_path)

    if not validator.initialize():
        logger.error("Model initialization failed")
        return

    result = validator.run_inference(
        image_path=image_path,
        task="detection",
        categories="person"
    )

    logger.info(f"Raw output: {result['raw_output'][:500]}...")

    validator.visualize_results(result, output_path)

    logger.info("Validation completed successfully")


if __name__ == "__main__":
    main()
