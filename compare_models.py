import argparse
import datetime
from pathlib import Path

import torch

from model import build_model
from train_din import build_paths, evaluate, make_iterator
from utils import calc_rela_impr, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DIN against Wide&Deep and compute RelaImpr.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=18)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model_result(model_name, paths, args, device):
    train_iterator = make_iterator(paths["train_file"], paths, args)
    n_uid, n_mid, n_cat = train_iterator.get_n()
    model = build_model(model_name, n_uid, n_mid, n_cat, embedding_dim=args.embedding_dim).to(device)
    ckpt_path = paths["ckpt_dir"] / ("%s_best_seed%s.pt" % (model_name.lower(), args.seed))#
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found for %s: %s" % (model_name, ckpt_path))

    checkpoint = torch.load(ckpt_path, map_location=device)#找到最优模型的 checkpoint 文件，并加载模型参数到 model 中
    model.load_state_dict(checkpoint["model_state_dict"])
    result = evaluate(model, paths["test_file"], paths, args, device)
    result["checkpoint"] = str(ckpt_path)
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    paths = build_paths(args)

    wide_deep_result = load_model_result("WIDE_DEEP", paths, args, device)
    din_result = load_model_result("DIN", paths, args, device)
    if wide_deep_result["auc"] > 0:
        auc_growth_percent = (din_result["auc"] - wide_deep_result["auc"]) / wide_deep_result["auc"] * 100.0
    else:
        auc_growth_percent = None
    rela_impr = calc_rela_impr(din_result["auc"], wide_deep_result["auc"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_path = paths["compare_dir"] / ("compare_seed%s_%s.json" % (args.seed, timestamp))
    payload = {
        "seed": args.seed,
        "device": str(device),
        "baseline_model": "WIDE_DEEP",
        "measured_model": "DIN",
        "wide_deep": wide_deep_result,
        "din": din_result,
        "auc_growth_percent": auc_growth_percent,
        "rela_impr_percent": rela_impr,
    }
    save_json(compare_path, payload)

    print("Wide&Deep ---- auc: %.4f ---- loss: %.4f ---- accuracy: %.4f" % (
        wide_deep_result["auc"], wide_deep_result["loss"], wide_deep_result["accuracy"]
    ))
    print("DIN --------- auc: %.4f ---- loss: %.4f ---- accuracy: %.4f" % (
        din_result["auc"], din_result["loss"], din_result["accuracy"]
    ))
    if auc_growth_percent is None:
        print("AUC Growth(DIN vs Wide&Deep): N/A (baseline auc is 0)")
    else:
        print("AUC Growth(DIN vs Wide&Deep): %.2f%%" % auc_growth_percent)
    print("RelaImpr(DIN vs Wide&Deep): %.2f%%" % rela_impr)
    print("comparison_file: %s" % compare_path)


if __name__ == "__main__":
    main()
