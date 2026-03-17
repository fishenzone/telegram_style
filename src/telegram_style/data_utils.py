import random
from datasets import Dataset
from .prompts import make_user_prompt

from .config import (
    TRAIN_SIZE,
    SEED,
    CHANNELS,
    INPUTS_TYPE1_PATH,
    INPUTS_TYPE2_PATH,
    REFERENCE_TYPE1_PATH,
    REFERENCE_TYPE2_PATH,
)
from .io_utils import load_lines, save_lines, save_jsonl, load_jsonl


def split_data(inputs, outputs, train_size=TRAIN_SIZE, seed=SEED, to_split=11):
    assert len(inputs) == len(outputs), "inputs and outputs must have same length"
    idx = list(range(len(inputs)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    train = [{"input": inputs[i], "output": outputs[i]} for i in train_idx]
    test = [{"input": inputs[i], "output": outputs[i]} for i in test_idx]
    return train, test[:to_split]


def prepare_data_splits():
    inputs_banki = load_lines(CHANNELS["type1"]["raw_inputs"])
    outputs_banki = load_lines(CHANNELS["type1"]["raw_outputs"])

    inputs_moscow = load_lines(CHANNELS["type2"]["raw_inputs"])
    outputs_moscow = load_lines(CHANNELS["type2"]["raw_outputs"])

    assert len(inputs_banki) == len(outputs_banki), "banki_oil input/output length mismatch"
    assert len(inputs_moscow) == len(outputs_moscow), "moscowach input/output length mismatch"

    train_banki, test_banki = split_data(inputs_banki, outputs_banki)
    train_moscow, test_moscow = split_data(inputs_moscow, outputs_moscow)

    save_lines([x["input"] for x in test_banki], INPUTS_TYPE1_PATH)
    save_lines([x["input"] for x in test_moscow], INPUTS_TYPE2_PATH)

    save_lines([x["output"] for x in test_banki], REFERENCE_TYPE1_PATH)
    save_lines([x["output"] for x in test_moscow], REFERENCE_TYPE2_PATH)

    save_jsonl(train_banki, CHANNELS["type1"]["train_pairs_path"])
    save_jsonl(test_banki, CHANNELS["type1"]["test_pairs_path"])

    save_jsonl(train_moscow, CHANNELS["type2"]["train_pairs_path"])
    save_jsonl(test_moscow, CHANNELS["type2"]["test_pairs_path"])

    return {
        "type1": {"train": train_banki, "test": test_banki},
        "type2": {"train": train_moscow, "test": test_moscow},
    }


def load_saved_splits():
    return {
        "type1": {
            "train": load_jsonl(CHANNELS["type1"]["train_pairs_path"]),
            "test": load_jsonl(CHANNELS["type1"]["test_pairs_path"]),
        },
        "type2": {
            "train": load_jsonl(CHANNELS["type2"]["train_pairs_path"]),
            "test": load_jsonl(CHANNELS["type2"]["test_pairs_path"]),
        },
    }


def build_chat_texts(pairs, system_prompt, tokenizer):
    texts = []
    for item in pairs:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": make_user_prompt(item["input"])},
            {"role": "assistant", "content": item["output"]},
        ]
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append(text)
    return texts


def build_dataset_from_pairs(pairs, system_prompt, tokenizer):
    texts = build_chat_texts(pairs, system_prompt, tokenizer)
    dataset = Dataset.from_dict({"text": texts})
    return dataset, texts
