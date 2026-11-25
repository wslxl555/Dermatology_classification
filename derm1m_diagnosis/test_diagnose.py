"""
derm1m_diagnosis 的冒烟测试 / 示例。

纯 Python，不依赖 torch。直接运行:
    python derm1m_diagnosis/test_diagnose.py

用若干典型「皮损计数 → 预期病种」病例验证本体推理的排序是否临床合理。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from derm1m_diagnosis import diagnose, format_report, normalize_counts


def check(label, counts, expected_top):
    result = diagnose(counts)
    top = result["top_disease"]
    ok = (top == expected_top)
    mark = "PASS" if ok else "FAIL"
    print(f"\n{'=' * 60}\n[{mark}] {label}\n{'=' * 60}")
    print(format_report(result))
    if not ok:
        print(f"  !! 期望 top={expected_top}, 实际 top={top}")
    return ok


def main():
    cases = [
        # 大疱 ×3 + 糜烂 + 痂 → 类天疱疹
        ("典型类天疱疹（大疱为主）", {"bulla": 3, "erosion": 1, "crust": 1},
         "bullous_pemphigoid"),
        # 丘疹 + 斑块 + 红斑、无鳞屑/结痂 → 荨麻疹（风团样）
        ("典型荨麻疹（风团样、无鳞屑结痂）",
         {"plaque": 2, "papule": 1, "macule": 1}, "urticaria"),
        # 鳞屑 + 水疱 + 丘疹 + 糜烂 → 湿疹
        ("典型湿疹（多形性损害组合）",
         {"scale": 3, "vesicle": 2, "papule": 1, "erosion": 1}, "eczema"),
        # 水疱 + 糜烂（渗出）+ 红斑，无大疱 → 接触性皮炎（急性）
        ("急性接触性皮炎（水疱/糜烂/红斑）",
         {"vesicle": 3, "erosion": 2, "macule": 2}, "contact_dermatitis"),
        # 鳞屑 + 结痂 → 不应判荨麻疹（荨麻疹反对鳞屑/结痂）
        ("鳞屑+结痂（排除荨麻疹）",
         {"scale": 2, "crust": 1}, "eczema"),
        # 慢性湿疹（苔藓化 + 抓痕 + 鳞屑）→ 湿疹
        ("慢性湿疹（苔藓化/抓痕/鳞屑）",
         {"lichenification": 2, "excoriation": 1, "scale": 2}, "eczema"),
    ]

    all_ok = True
    for label, counts, expected in cases:
        all_ok &= check(label, counts, expected)

    # 证据不足场景
    print(f"\n{'=' * 60}\n[INFO] 空输入（证据不足）\n{'=' * 60}")
    print(format_report(diagnose({})))

    # 归一化测试：中文 / Derm1M 概念名
    norm = normalize_counts(["大疱", "大疱", "wheal", "scaly", "erosion"])
    print(f"\n归一化测试: {norm}  (期望 bulla=2, plaque=1, scale=1, erosion=1)")

    print(f"\n{'=' * 60}\n结果: {'全部通过' if all_ok else '存在失败用例'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
