import transformers
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


def main():
    model_path = "./models/janus-7b"
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,  trust_remote_code=True)


if __name__ == "__main__":
    main()
