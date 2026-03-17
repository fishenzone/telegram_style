import torch
from unsloth import FastLanguageModel
from .prompts import make_user_prompt

from .config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4BIT,
    LOAD_IN_8BIT,
    FULL_FINETUNING,
    LORA_R,
    LORA_ALPHA,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GEN_TOP_K,
)
from .io_utils import sanitize_generation
from .memory_utils import cleanup, print_gpu


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        full_finetuning=FULL_FINETUNING,
    )
    return model, tokenizer


def attach_lora(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def unload_and_attach_fresh_lora(model):
    print("Before unload:")
    print_gpu()

    cleanup()

    model = model.unload()

    cleanup()

    print("After unload:")
    print_gpu()

    model = attach_lora(model)

    cleanup()

    print("Fresh LoRA attached:")
    print_gpu()
    return model


def generate_texts(model, tokenizer, inputs, system_prompt, max_new_tokens):
    FastLanguageModel.for_inference(model)
    results = []

    for i, text in enumerate(inputs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": make_user_prompt(text)},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs_tok = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs_tok,
                max_new_tokens=max_new_tokens,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                top_k=GEN_TOP_K,
                do_sample=True,
            )

        new_tokens = output[0][inputs_tok["input_ids"].shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)
        result = sanitize_generation(result)
        results.append(result)
        print(f"[{i+1}/{len(inputs)}] done")

    return results
