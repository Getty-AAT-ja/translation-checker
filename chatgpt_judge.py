import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

"""
chatgpt_judge.py  
===================
This script reads *input.csv* and evaluates translation quality with an OpenAI model.
If *correct_label.csv* (no header, two columns: **URL**, **correct_label**) is present,
it overwrites the value in the ``元ラベル`` column by matching on the URL column
(default column name ``URL``).  

Key changes from the previous version
-------------------------------------
1. Added ``CORRECT_LABEL_PATH`` and merging logic to replace inaccurate labels.
2. Added ``URL_COLUMN_NAME`` constant for flexibility if your column is named
   something other than ``URL``.
3. Re‑structured the configuration section and removed the hard‑coded API key.
"""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_PATH = "Getty AAT-ja-2050425亀田さん作業用.csv"
CORRECT_LABEL_PATH = "correct_label.csv"  # No header, [URL, correct_label]
OUTPUT_PATH = "output_with_scores_4.csv"
LOG_PATH = "log.txt"

# Set this to your URL column name in both CSVs if it differs from "URL"
URL_COLUMN_NAME = "リソース"

# -----------------------------------------------------------------------------
# OpenAI client initialisation  (expects OPENAI_API_KEY in environment)
# -----------------------------------------------------------------------------
client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

# -----------------------------------------------------------------------------
# Load input data and, if available, overwrite 元ラベル using correct_label.csv
# -----------------------------------------------------------------------------

def load_and_correct_labels() -> pd.DataFrame:
    """Read *input.csv* and, if present, merge in corrected labels."""
    # Load main data
    df = pd.read_csv(INPUT_PATH)

    # Input validation
    if URL_COLUMN_NAME not in df.columns:
        raise KeyError(
            f"'{URL_COLUMN_NAME}' column is required in {INPUT_PATH} to match URLs"
        )

    # Try to load corrected labels
    if Path(CORRECT_LABEL_PATH).exists():
        df_correct = pd.read_csv(
            CORRECT_LABEL_PATH, header=None, names=[URL_COLUMN_NAME, "correct_label"]
        )
        # Merge on URL and overwrite ラベル原文 where a correction exists
        df = df.merge(df_correct, on=URL_COLUMN_NAME, how="left")
        if "ラベル原文" not in df.columns:
            # If ラベル原文 column didn’t exist, create it
            df["ラベル原文"] = df["correct_label"]
        else:
            df.loc[~df["correct_label"].isna(), "ラベル原文"] = df.loc[
                ~df["correct_label"].isna(), "correct_label"
            ]
        df.drop(columns=["correct_label"], inplace=True)
    else:
        print(
            f"Warning: {CORRECT_LABEL_PATH} not found. Proceeding without label correction."
        )

    # ✅ 文献1〜3すべてに非空の値がある行だけを残す（NaN + 空文字対応）
    required_columns = ["文献1", "文献2", "文献3"]
    for col in required_columns:
        df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
    return df


def build_prompt(row: pd.Series) -> str:
    """Compose the evaluation prompt for a single row."""
    return f"""
以下の専門用語辞書の翻訳情報について、8つの観点から5点満点で評価し、それぞれ理由を述べてください。

【元ラベル（英語）】:
{row['ラベル原文']}

【日本語ラベル（人訳）】:
{row['日本語ラベル(人訳)']}

【解説文（英語）】:
{row['解説原文']}

【解説文（人訳・修正）】:
{row['解説文(人訳)']}

評価項目:
(1) 情報の抜けが無いか。つまり、「解説文（英語）」にある情報がその訳「解説文（人訳・修正）」で抜け落ちてないか。なければ5点満点。
(2) 自然な日本語文であるか。「解説文（人訳・修正）」に対して評価を行ってください。
(3) 対象を理解するために十分な説明か。「解説文（英語）」に対して評価を行ってください。
(4) 関連深い専門知識を適切に用いて説明されているか（理由に関連する専門を列挙してください）。「解説文（英語）」に対して評価を行ってください。
(5) 事実ではない内容を含んでいないか。専門知識に基づいて判断してください。「解説文（人訳・修正）」に対して評価を行ってください。
(6) 元ラベルと日本語ラベルが同じ対象を指しているか。ズレ、つまりどちらかだけが含む対象があったり、指示対象の曖昧性はないか。
(7) ある分野でよく使われる用語・訳語がある場合、それと一致しているか。独自に造語していないか。「日本語ラベル（人訳）」に対して評価を行ってください。
(8) ある分野でよく使われる用語・訳語がある場合、それと一致しているか。独自に造語していないか。「解説文（人訳・修正）」に対して評価を行ってください。

形式は以下に従ってください：

(1) スコア: X/5 理由: ...
(2) スコア: X/5 理由: ...
...
(7) スコア: X/5 理由: ...
"""


def parse_response(text: str) -> dict:
    """Extract scores/reasons from the model response as a dict."""
    import re

    results = {}
    for i in range(1, 9):
        score_match = re.search(rf"\({i}\) スコア:\s*([0-5])/5", text)
        reason_match = re.search(
            rf"\({i}\) スコア:\s*[0-5]/5\s*理由:\s*(.+?)(?=\(\d+\) スコア:|$)",
            text,
            re.DOTALL,
        )
        results[f"Score_{i}"] = int(score_match.group(1)) if score_match else None
        results[f"Reason_{i}"] = (
            reason_match.group(1).strip() if reason_match else ""
        )
    return results


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def main():
    # 入力ファイル読み込み（フィルタ込み）
    df = load_and_correct_labels()
    print("🟦 入力データ（フィルタ済み）の読み込み完了")
    print(f"行数: {len(df)}")
    print("列名:", df.columns.tolist())
    print("先頭5行:\n", df.head())

    # 出力ファイルがあれば復元処理へ
    if Path(OUTPUT_PATH).exists():
        df_prev = pd.read_csv(OUTPUT_PATH)
        print(f"🟩 出力ファイル {OUTPUT_PATH} を読み込み（{len(df_prev)} 行）")
        print("出力ファイルの先頭5行:\n", df_prev.head())

        # 突き合わせ前に key 列（URL or リソース）確認
        print("df[URL_COLUMN_NAME] サンプル:", df[URL_COLUMN_NAME].head())
        print("df_prev[URL_COLUMN_NAME] サンプル:", df_prev[URL_COLUMN_NAME].head())

        # set_indexして上書き
        df_prev.set_index(URL_COLUMN_NAME, inplace=True)
        df.set_index(URL_COLUMN_NAME, inplace=True)

        for i in range(1, 9):
            score_col = f"Score_{i}"
            reason_col = f"Reason_{i}"
            if score_col in df_prev.columns:
                df[score_col] = df_prev[score_col]
            if reason_col in df_prev.columns:
                df[reason_col] = df_prev[reason_col]

        df.reset_index(inplace=True)
    else:
        print("🟥 出力ファイルが存在しないため新規処理を行います")
        for i in range(1, 9):
            df[f"Score_{i}"] = None
            df[f"Reason_{i}"] = ""

    # スコア列確認
    print("Score_1 の欠損行数:", df["Score_1"].isna().sum())

    incomplete_mask = df["Score_1"].isna()
    if not incomplete_mask.any():
        print("✅ すべての行が処理済みです。")
        return

    START_INDEX = incomplete_mask.idxmax()
    iloc_start = df.index.get_loc(START_INDEX)
    print(f"▶ 処理を {iloc_start} 行目（フィルタ後）から再開します。")

    # ChatGPT evaluation loop
    with open(LOG_PATH, "a", encoding="utf-8") as log:  # ← 追記モードに変更
        for idx, row in df.iloc[iloc_start:].iterrows():
            print(f"[{idx}] 処理中ラベル: {row['ラベル原文']}")
            prompt = build_prompt(row)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # adjust model as needed
                    messages=[
                        {
                            "role": "system",
                            "content": "あなたは英語と日本語が堪能な翻訳のレビュアーで、芸術を中心として様々な専門知識を持っています。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2000,
                )
                output = response.choices[0].message.content
                log.write(f"===== Index: {idx} =====\n{output}\n\n")

                parsed = parse_response(output)
                for key, value in parsed.items():
                    df.at[idx, key] = value

                # ✅ 各行ごとに保存（中断対策）
                df.to_csv(OUTPUT_PATH, index=False)

                time.sleep(0.5)  # polite rate limit
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue

if __name__ == "__main__":
    main()