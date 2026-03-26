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


def print_model_structure(model, output_file=None):
    logger.info("=" * 100)
    logger.info("模型算子级别子模块结构")
    logger.info("=" * 100)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"总参数量: {total_params:,}")
    logger.info("-" * 100)

    module_list = []
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            module_type = type(module).__name__
            module_list.append({
                "name": name,
                "type": module_type,
                "params": params,
                "params_M": params / 1_000_000
            })
            logger.info(f"{name} | {module_type} | 参数量: {params:,} ({params/1_000_000:.2f}M)")

    logger.info("=" * 100)
    logger.info("按模块类型统计:")
    logger.info("-" * 100)

    type_stats = {}
    for m in module_list:
        t = m["type"]
        if t not in type_stats:
            type_stats[t] = {"count": 0, "params": 0}
        type_stats[t]["count"] += 1
        type_stats[t]["params"] += m["params"]

    for module_type, stats in sorted(type_stats.items(), key=lambda x: -x[1]["params"]):
        logger.info(f"  {module_type}: 数量={stats['count']}, 参数量={stats['params']:,} ({stats['params']/1_000_000:.2f}M)")

    logger.info("=" * 100)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("模型算子级别子模块结构\n")
            f.write("=" * 100 + "\n")
            f.write(f"总参数量: {total_params:,}\n")
            f.write("=" * 100 + "\n")
            for m in module_list:
                f.write(f"{m['name']} | {m['type']} | 参数量: {m['params']:,} ({m['params_M']:.2f}M)\n")
            f.write("=" * 100 + "\n")
            f.write("按模块类型统计:\n")
            f.write("-" * 100 + "\n")
            for module_type, stats in sorted(type_stats.items(), key=lambda x: -x[1]["params"]):
                f.write(f"  {module_type}: 数量={stats['count']}, 参数量={stats['params']:,} ({stats['params']/1_000_000:.2f}M)\n")

        logger.info(f"结构已保存到: {output_file}")

    return module_list


def main():
    model_path = "/opt/data/models/IDEA-Research/Rex-Omni"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./model_structure_{timestamp}.txt"

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

    print_model_structure(model_exec, output_file=output_file)

    logger.info("模型结构打印完成!")


if __name__ == "__main__":
    main()
