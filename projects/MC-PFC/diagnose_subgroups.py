"""子集诊断：录音级（按受试者分组）vs 受试者级 5 折 CV。

用于解释为什么 `AD_F_*`/`HC_F_*` 这批多录音受试者在受试者级聚合后信号变弱。
录音级 CV 必须用 **GroupKFold（按受试者分组）**，否则同一受试者的不同 session
会同时出现在训练/测试折，AUC 会被严重高估。

运行：
    python diagnose_subgroups.py
输出：
    logs/diagnostics/SUBGROUP.md
    logs/diagnostics/subgroup_cv.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import common
import run_mcpfc as R


def cv_predict(X, y, groups, splitter, standardize):
    p = np.zeros(len(y), dtype=float)
    for tr, va in splitter.split(X, y, groups):
        Xtr, Xva = X[tr], X[va]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
        clf = SVC(
            C=config.SVM_C, kernel="rbf", gamma=config.SVM_GAMMA,
            class_weight="balanced", probability=True, random_state=0,
        ).fit(Xtr, y[tr])
        p[va] = clf.predict_proba(Xva)[:, 1]
    return float(roc_auc_score(y, p))


def main():
    X_en, y_en, _ = common.load_english_features()
    scaler, keep, _, _, _ = R.fit_source_model(X_en, y_en, config.N_FEATURES)

    records = common.load_chinese_records()
    subjects = common.chinese_subjects()
    X_rec = common.load_record_matrix("zh", records.uid)
    Z_rec = scaler.transform(X_rec)[:, keep]
    y_rec = records.label.to_numpy()
    g_rec = np.array([common.subject_of(u) for u in records.uid])
    is_num_rec = records.uid.str.isdigit().to_numpy()

    X_sub, y_sub, names, _ = common.load_chinese_features()
    Z_sub = scaler.transform(X_sub)[:, keep]
    is_num_sub = np.array([s.isdigit() for s in names])

    rows = []
    for name, m_rec, m_sub in [
        ("numeric80", is_num_rec, is_num_sub),
        ("ADF70", ~is_num_rec, ~is_num_sub),
    ]:
        for tag, Xr, Xs, std in [
            ("raw1024", X_rec[m_rec], X_sub[m_sub], True),
            ("en60", Z_rec[m_rec], Z_sub[m_sub], False),
        ]:
            rec_auc = cv_predict(
                Xr, y_rec[m_rec], g_rec[m_rec], GroupKFold(n_splits=5), standardize=std
            )
            sub_auc = cv_predict(
                Xs, y_sub[m_sub], None,
                StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                standardize=std,
            )
            rows.append(
                {
                    "subset": name,
                    "n_subjects": int(m_sub.sum()),
                    "n_records": int(m_rec.sum()),
                    "features": tag,
                    "record_gkf_auc": rec_auc,
                    "subject_cv_auc": sub_auc,
                }
            )
    df = pd.DataFrame(rows)
    out = config.LOG_DIR / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "subgroup_cv.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 子集诊断：录音级（GroupKFold，按受试者分组）vs 受试者级\n",
        "| 子集 | 受试者 | 录音 | 特征 | 录音级 GroupKFold AUC | 受试者级 5 折 AUC |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.subset} | {r.n_subjects} | {r.n_records} | {r.features} | "
            f"{r.record_gkf_auc:.3f} | {r.subject_cv_auc:.3f} |"
        )
    lines.append("")
    lines.append(
        "> 录音级 CV 按受试者分组，避免同一人的不同 session 跨折泄露；"
        "受试者级 CV 对每个受试者只有一条（均值聚合后）样本。"
    )
    (out / "SUBGROUP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\n完成：{out}")


if __name__ == "__main__":
    main()
