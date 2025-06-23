from vllm import LLMEngine, PoolingParams, EngineArgs
import torch


def main():
    engine_args = EngineArgs(model="./models/orpheus", dtype=torch.bfloat16, cpu_offload_gb=5.0, max_model_len=13107)
    engine = LLMEngine.from_engine_args(engine_args)
    message = "Hello world"
    text = engine.encode(message, PoolingParams(), 0)
    print(text)
    
    return


if __name__ == "__main__":
    main()
