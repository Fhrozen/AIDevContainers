import sys
import os

sys.path.append(os.path.abspath("./Spark-TTS"))

from sparktts.models.bicodec import BiCodec
from sparktts.utils.file import load_config

import torch


def main():
    model = BiCodec.load_from_checkpoint("./models/spark_tts/BiCodec")
    model.forward = model.detokenize
    
    semantic_tokens = torch.randint(1, 1000, [1, 100])
    global_tokens = torch.randint(1, 1000, [1, 1, 32])
    outputs = model(semantic_tokens, global_tokens)
    print(outputs.shape)
    torch.onnx.export(
        model,                                     # model to export
        (semantic_tokens, global_tokens),                   # model inputs
        "./models_converted/spark_tts/bicodec.onnx",                      # output file
        export_params=True,                          # store trained parameters
        opset_version=18,                            # ONNX version
        do_constant_folding=True,                    # execute constant folding
        input_names=['semantic_tokens', 'global_tokens'],   # input tensor names
        output_names=['audio'],                      # output tensor names
        dynamic_axes={                               # variable length axes
            'semantic_tokens': {0: 'batch_size', 1: "length"},
            'global_tokens': {0: 'batch_size'}
        }
    )
    
    return


if __name__ == "__main__":
    main()
