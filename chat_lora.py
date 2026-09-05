import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_ADAPTER = "./qwen3-lora"


def load_lora_model():
    print("Loading base model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER,
    )

    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages):

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[
        0,
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def chat(model, tokenizer):

    print("\nLoRA Qwen3 chat")
    print("Type 'exit' to quit.\n")

    messages = []

    while True:

        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        messages.append({
            "role": "user",
            "content": user_message,
        })

        answer = generate(
            model,
            tokenizer,
            messages,
        )

        print(f"Assistant: {answer}\n")

        messages.append({
            "role": "assistant",
            "content": answer,
        })


def main():

    model, tokenizer = load_lora_model()

    chat(model, tokenizer)


if __name__ == "__main__":
    main()