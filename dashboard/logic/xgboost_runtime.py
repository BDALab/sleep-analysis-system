import os
import platform
import shutil


def xgboost_device():
    requested = os.environ.get("GENEACTIV_XGB_DEVICE", "auto").strip().lower()
    if requested == "auto":
        if platform.system() == "Darwin":
            return "cpu"
        return "cuda" if shutil.which("nvidia-smi") else "cpu"

    if requested == "cpu" or requested.startswith("cuda"):
        if requested.startswith("cuda") and platform.system() == "Darwin":
            raise ValueError(
                "CUDA XGBoost is unavailable on macOS. "
                "Use GENEACTIV_XGB_DEVICE=cpu."
            )
        return requested
    raise ValueError(
        "GENEACTIV_XGB_DEVICE must be auto, cpu, cuda, or cuda:<ordinal>"
    )


def xgboost_threads():
    requested = os.environ.get("GENEACTIV_XGB_THREADS")
    if requested:
        threads = int(requested)
        if threads < 1:
            raise ValueError("GENEACTIV_XGB_THREADS must be at least 1")
        return threads

    cpu_count = os.cpu_count() or 1
    # Leave some capacity for macOS and avoid nested-CV oversubscription.
    return max(1, min(8, cpu_count - 2 if cpu_count > 4 else cpu_count))


def configure_xgboost_params(params=None):
    configured = dict(params or {})
    if "seed" in configured and "random_state" not in configured:
        configured["random_state"] = configured.pop("seed")

    for obsolete in ("gpu_id", "predictor", "use_label_encoder"):
        configured.pop(obsolete, None)

    configured["device"] = xgboost_device()
    configured["tree_method"] = os.environ.get(
        "GENEACTIV_XGB_TREE_METHOD",
        "hist",
    )
    configured["n_jobs"] = xgboost_threads()
    configured.setdefault("max_bin", 256)
    return configured


def configure_xgboost_core_params(params=None):
    configured = configure_xgboost_params(params)
    configured["nthread"] = configured.pop("n_jobs")
    return configured


def cpu_fallback_params(params=None):
    configured = dict(params or {})
    configured["device"] = "cpu"
    configured["tree_method"] = "hist"
    configured["n_jobs"] = xgboost_threads()
    return configured


def is_xgboost_device_error(exc):
    message = str(exc).lower()
    return any(
        keyword in message
        for keyword in ("cuda", "gpu", "device", "libomp", "openmp")
    )


def xgboost_runtime_metadata():
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": xgboost_device(),
        "tree_method": os.environ.get("GENEACTIV_XGB_TREE_METHOD", "hist"),
        "threads": xgboost_threads(),
    }
