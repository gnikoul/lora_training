import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
DATA_FILE = Path("train.jsonl")
ADAPTER_DIR = Path("qwen3-lora")

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


@torch.inference_mode()
def generate(model, tokenizer, message):

    messages = [
        {
            "role": "user",
            "content": message,
        }
    ]

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
        max_new_tokens=100,
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
    
    
prompt = "Where is Greece?"
print("========== BASE ==========")
print(generate(model, tokenizer, prompt))


def add_lora(model, rank=1, alpha=16, dropout=0.05):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

model = add_lora(model)



class InstructionDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=256):
        self.examples = []

        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        item = self.examples[index]

        user_messages = [
            {
                "role": "user",
                "content": item["instruction"],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": item["instruction"],
            },
            {
                "role": "assistant",
                "content": item["response"],
            },
        ]

        # User prompt + assistant generation prompt
        prompt = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        )

        # User prompt + actual assistant response
        full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            enable_thinking=False,
        )

        prompt_ids = prompt["input_ids"]
        full_ids = full["input_ids"]

        # They may still be tensors depending on the Transformers version.
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()

        if isinstance(full_ids, torch.Tensor):
            full_ids = full_ids.tolist()

        # Truncate
        full_ids = full_ids[:self.max_length]

        # Ignore user tokens in the loss
        labels = [-100] * len(full_ids)

        response_start = min(len(prompt_ids), len(full_ids))

        labels[response_start:] = full_ids[response_start:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }
    
dataset = InstructionDataset(
    DATA_FILE,
    tokenizer
)

def train_lora(
    model,
    tokenizer,
    dataset,
    output_dir=ADAPTER_DIR,
):

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),

        num_train_epochs=10,

        per_device_train_batch_size=2,

        learning_rate=2e-4,

        bf16=True,

        logging_steps=1,

        save_strategy="no",

        report_to="none",

        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model

train_lora(
    model,
    tokenizer,
    dataset,
)

tests = [
    "hello",
    "What is a cat?",
    "What is a computer?",
    "Where is Athens?",
    "What is an elephant?",
    "What is quantum mechanics?",
    "Tell me about the Moon.",
    "Who was Albert Einstein?",
    "What is 17 * 3?",
]

print("========== LORA ==========")
for prompt in tests:
    print(f"\nPROMPT: {prompt}")
    print(f"ANSWER: {generate(model, tokenizer, prompt)}")
    
    
