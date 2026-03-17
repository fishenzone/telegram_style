import torch
from trl import SFTTrainer, SFTConfig

from .config import NUM_EPOCHS


def build_trainer(model, tokenizer, dataset, output_dir):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            padding_free=False,
            output_dir=str(output_dir),
            save_strategy="no",
        ),
    )
    return trainer


def train_and_save(model, tokenizer, dataset, output_dir, label):
    trainer = build_trainer(model, tokenizer, dataset, output_dir)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB reserved before training {label}.")

    print(f"=== Training {label} ===")
    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{label} training runtime: {trainer_stats.metrics['train_runtime']:.2f} sec")
    print(f"Peak reserved memory = {used_memory} GB")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB")
    print(f"Peak reserved memory % = {used_percentage}%")
    print(f"Peak reserved memory for training % = {lora_percentage}%")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA to {output_dir}")

    return trainer, trainer_stats
