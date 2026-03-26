from vllm import LLM
import torch
import logging
import os
from datetime import datetime

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    elif hasattr(torch, "mlu") and torch.mlu.is_available():
        return "mlu"
    else:
        return "cpu"


def export_to_onnx(model, output_path="./model.onnx", seq_len=512):
    logger.info("=" * 100)
    logger.info("开始导出 ONNX 模型...")
    logger.info("=" * 100)

    device = get_device()
    logger.info(f"检测到设备: {device}")
    logger.info("注意: ONNX 导出将在 CPU 上进行，以确保兼容性")

    model.eval()
    model = model.cpu()

    dummy_input = torch.zeros(seq_len, dtype=torch.long)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": {0: "sequence_length"},
                    "output": {0: "sequence_length"}
                }
            )
        logger.info(f"ONNX 模型已保存到: {output_path}")
        logger.info("可以使用 Netron 打开查看: https://netron.app/")
        return True
    except Exception as e:
        logger.error(f"ONNX 导出失败: {e}")
        logger.warning("大型 LLM 模型导出 ONNX 可能需要特殊处理，建议使用 ray/DeepSpeed 等工具")
        return False


def export_submodule_to_onnx(model, submodule_name, output_path="./submodule.onnx", seq_len=512):
    logger.info("=" * 100)
    logger.info(f"尝试导出子模块: {submodule_name}")
    logger.info("=" * 100)

    device = get_device()
    logger.info(f"检测到运行时设备: {device}")
    logger.info("注意: ONNX 导出将在 CPU 上进行，以确保兼容性")

    for name, module in model.named_modules():
        if submodule_name in name:
            logger.info(f"找到子模块: {name} | 类型: {type(module).__name__}")

            module.eval()
            module = module.cpu()
            dummy_input = torch.zeros(1, seq_len, 4096, dtype=torch.float16)

            try:
                with torch.no_grad():
                    torch.onnx.export(
                        module,
                        dummy_input,
                        output_path,
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=["hidden_states"],
                        output_names=["output"],
                        dynamic_axes={
                            "hidden_states": {0: "batch", 1: "sequence_length", 2: "hidden_size"},
                            "output": {0: "batch", 1: "sequence_length", 2: "hidden_size"}
                        }
                    )
                logger.info(f"子模块 ONNX 已保存到: {output_path}")
                return True
            except Exception as e:
                logger.error(f"子模块导出失败: {e}")
                return False

    logger.warning(f"未找到子模块: {submodule_name}")
    return False


def main():
    model_path = "/opt/data/models/IDEA-Research/Rex-Omni"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./onnx_exports"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("开始加载 vLLM 模型...")

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

    logger.info("模型加载完成，获取模型执行对象...")

    model_exec = model.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()

    logger.info("=" * 100)
    logger.info("开始导出完整模型到 ONNX...")
    logger.info("=" * 100)

    onnx_file = os.path.join(output_dir, f"model_{timestamp}.onnx")
    success = export_to_onnx(model_exec, output_path=onnx_file, seq_len=512)

    if not success:
        logger.warning("完整模型导出失败，尝试导出 Embedding 层...")
        embed_file = os.path.join(output_dir, f"embedding_{timestamp}.onnx")
        export_submodule_to_onnx(model_exec, "model.embed_tokens", output_path=embed_file)

        logger.warning("尝试导出单个 Transformer 层...")
        layer_file = os.path.join(output_dir, f"transformer_layer_{timestamp}.onnx")
        export_submodule_to_onnx(model_exec, "model.layers.0", output_path=layer_file, seq_len=512)

    logger.info("=" * 100)
    logger.info("ONNX 导出完成!")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
