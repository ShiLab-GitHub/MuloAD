import os
import argparse
import copy
import pandas as pd
from feat_importance import cal_feat_imp, summarize_imp_feat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="ROSMAP")
    parser.add_argument("--views", type=str, default="1,2,3")
    parser.add_argument("--num_class", type=int, default=None)
    parser.add_argument("--topn", type=int, default=30)
    return parser.parse_args()


def get_omics_labels(data_folder, view_list):
    if data_folder == "RiMod_FTD":
        return ["RNA", "Methylation", "CAGE"]
    if data_folder == "HD":
        return ["RNA"]
    if data_folder == "ROSMAP":
        return ["RNA", "Methylation", "miRNA"]
    if data_folder == "BRCA":
        return ["RNA", "miRNA"]
    return [str(v) for v in view_list]


def infer_num_class(data_folder, explicit_num_class):
    if explicit_num_class is not None:
        return explicit_num_class
    if data_folder in ["ROSMAP", "RiMod_FTD", "HD"]:
        return 2
    if data_folder == "BRCA":
        return 5
    return 2


def find_replicate_dirs(model_root):
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"Model folder not found: {model_root}")
    digit_dirs = []
    for name in sorted(os.listdir(model_root)):
        path = os.path.join(model_root, name)
        if os.path.isdir(path) and name.isdigit():
            digit_dirs.append(name)
    if digit_dirs:
        return digit_dirs
    final_dir = os.path.join(model_root, "final")
    if os.path.isdir(final_dir):
        return ["final"]
    raise FileNotFoundError(f"No model weights found in {model_root}")


def main():
    args = parse_args()
    data_folder = args.data_folder
    view_list = [int(x) for x in args.views.split(",") if x]
    num_class = infer_num_class(data_folder, args.num_class)

    model_root = os.path.join(data_folder, "models")
    replicate_dirs = find_replicate_dirs(model_root)

    all_feat_importances = []
    for rep in replicate_dirs:
        model_dir = os.path.join(model_root, str(rep))
        featimp_list = cal_feat_imp(data_folder, model_dir, view_list, num_class)
        all_feat_importances.append(copy.deepcopy(featimp_list))

    df_top = summarize_imp_feat(all_feat_importances, topn=args.topn)
    df_top = df_top.copy()
    df_top["rank"] = range(1, len(df_top) + 1)

    out_dir = os.path.join(data_folder, "biomarkers")
    os.makedirs(out_dir, exist_ok=True)

    labels = get_omics_labels(data_folder, view_list)
    df_top["omics_label"] = df_top["omics"].map(
        lambda i: labels[int(i)] if int(i) < len(labels) else str(i)
    )
    df_top[["rank", "feat_name", "omics", "imp", "omics_label"]].to_csv(
        os.path.join(out_dir, f"biomarkers_top{args.topn}.csv"), index=False
    )

    for view_index in sorted(df_top["omics"].unique()):
        df_view = df_top[df_top["omics"] == view_index].copy()
        df_view["rank"] = range(1, len(df_view) + 1)
        df_view[["rank", "feat_name", "omics", "imp"]].to_csv(
            os.path.join(out_dir, f"{int(view_index) + 1}_top{args.topn}.csv"), index=False
        )


if __name__ == "__main__":
    main()
