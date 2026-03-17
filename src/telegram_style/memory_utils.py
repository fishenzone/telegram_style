import gc
import torch


def print_gpu():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def nuclear_cleanup(namespace=None):
    if namespace is not None:
        for name in list(namespace.keys()):
            if name.startswith("model") or name.startswith("trainer") or name.startswith("tokenizer"):
                del namespace[name]

    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:
        pass

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    print_gpu()


def drop_vars(namespace, names):
    for name in names:
        if name in namespace:
            del namespace[name]

    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
