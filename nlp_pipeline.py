# nlp_pipeline.py
"""
Hybrid NLP engine:
1. Custom PyTorch intent classifier  → “show_chart” or “answer_question”
2. Custom TensorFlow NER             → pull out columns / aggregates
3. LangChain RAG (FAISS + LLM)       → precise aggregation + chart spec
4. Matplotlib/Seaborn                → turn pandas → PNG → base64
"""

import re
import json
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Load the two home-grown models (trained from scratch) -----
from models.intent_model import load_intent_model
from models.qa_model import load_qa_model
from rag.chain import rag_answer   # LangChain RAG chain

# Load once when the module is imported
intent_model, intent_vocab, intent_label_to_id = load_intent_model()
qa_model, qa_word_to_id, qa_tag_to_id = load_qa_model()
id_to_tag = {v: k for k, v in qa_tag_to_id.items()}

# ------------------------------------------------------------------
# 1. Intent classification (PyTorch CNN)
# ------------------------------------------------------------------
def classify_intent(user_text: str) -> str:
    """Return one of: 'show_chart', 'answer_question', 'unknown'."""
    token_ids = intent_vocab.encode(user_text)
    input_tensor = torch.tensor([token_ids])               # shape (1, seq_len)

    with torch.no_grad():
        logits = intent_model(input_tensor)
        predicted_id = logits.argmax(dim=1).item()

    # map numeric id back to human label
    return [lbl for lbl, idx in intent_label_to_id.items() if idx == predicted_id][0]

# ------------------------------------------------------------------
# 2. Entity extraction (TensorFlow Bi-LSTM NER)
# ------------------------------------------------------------------
def extract_columns_and_aggregates(user_text: str) -> dict:
    """Return {'columns': [...], 'aggregate': 'sum' | 'avg' | ...}."""
    tokens = re.findall(r"\w+", user_text.lower())
    # pad / truncate to the length the model expects
    max_len = 20
    ids = [qa_word_to_id.get(t, 1) for t in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    batch = np.array([ids])

    predictions = qa_model.predict(batch, verbose=0)[0]
    tags = [id_to_tag[np.argmax(p)] for p in predictions[: len(tokens)]]

    columns, aggregate = [], None
    current_phrase = ""

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_phrase:
                # finish previous phrase
                columns.append(current_phrase) if "COL" in tag else aggregate = current_phrase
            current_phrase = token
        elif tag.startswith("I-"):
            current_phrase += " " + token
        else:
            if current_phrase:
                columns.append(current_phrase) if "COL" in tag else aggregate = current_phrase
            current_phrase = ""

    # flush the last phrase
    if current_phrase:
        # simple heuristic – if we never saw a tag, treat as column
        columns.append(current_phrase)

    return {"columns": columns, "aggregate": aggregate or "sum"}

# ------------------------------------------------------------------
# 3. Chart rendering (pure pandas + seaborn)
# ------------------------------------------------------------------
def render_chart(aggregated_df: pd.DataFrame, chart_spec: dict) -> str:
    """Return a base64 data-uri ready for <img src=...>."""
    plt.figure(figsize=(6, 4))
    chart_type = chart_spec["type"]

    if chart_type == "bar":
        sns.barplot(data=aggregated_df,
                    x=chart_spec["x"],
                    y=chart_spec["y"],
                    hue=chart_spec.get("hue"))
    elif chart_type == "line":
        sns.lineplot(data=aggregated_df,
                     x=chart_spec["x"],
                     y=chart_spec["y"],
                     hue=chart_spec.get("hue"))
    elif chart_type == "pie":
        series = aggregated_df.groupby(chart_spec["x"])[chart_spec["y"]].sum()
        plt.pie(series, labels=series.index, autopct="%1.1f%%")
    elif chart_type == "heatmap":
        pivot = aggregated_df.pivot_table(
            index=chart_spec["x"],
            columns=chart_spec.get("hue"),
            values=chart_spec["y"],
            aggfunc="sum",
        )
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=120)
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    return f"data:image/png;base64,{img_b64}"

# ------------------------------------------------------------------
# 4. Hybrid entry point – decides which path to take
# ------------------------------------------------------------------
def process_query_hybrid(user_message: str, full_dataframe: pd.DataFrame) -> str:
    """
    1. Intent → chart vs plain answer
    2. If chart → ask LangChain RAG for exact spec + optional natural language
    3. If plain → fall back to the original custom QA path
    """
    intent = classify_intent(user_message)

    # -------------------------------------------------
    # A. Chart path (LangChain RAG)
    # -------------------------------------------------
    if intent == "show_chart":
        rag_result = rag_answer(user_message)

        # RAG sometimes returns just an answer, sometimes a chart dict
        if "chart" in rag_result:
            spec = rag_result["chart"]

            # Normalise chart-type names the LLM might give
            type_map = {
                "bar chart": "bar",
                "line chart": "line",
                "pie chart": "pie",
                "heatmap": "heatmap",
            }
            spec["type"] = type_map.get(spec.get("type", "").lower(), "bar")

            # Build a tiny dataframe from the rows the retriever gave us
            retrieved_rows = rag_result.get("retrieved", [])
            sub_df = pd.DataFrame(retrieved_rows) if retrieved_rows else full_dataframe

            # Simple aggregation fallback
            group_col = spec.get("x") or sub_df.columns[0]
            value_col = spec.get("y") or sub_df.select_dtypes(include="number").columns[0]
            aggregated = sub_df.groupby(group_col)[value_col].sum().reset_index()

            spec.update({"x": group_col, "y": value_col})
            chart_html = render_chart(aggregated, spec)

            natural_answer = rag_result.get("answer", "")
            return f"{natural_answer}<br><img src='{chart_html}' style='max-width:100%'/>"

        # No chart spec → just return the textual answer
        return rag_result.get("answer", "I couldn’t create a chart.")

    # -------------------------------------------------
    # B. Plain question path (original custom logic)
    # -------------------------------------------------
    else:
        # Re-use the older pure-custom function for brevity
        return legacy_custom_answer(user_message, full_dataframe)

# ------------------------------------------------------------------
# Legacy custom QA (kept for non-chart questions)
# ------------------------------------------------------------------
def legacy_custom_answer(user_message: str, df: pd.DataFrame) -> str:
    """Very small fallback – total sales, etc."""
    if "total sales" in user_message.lower():
        total = df["sales"].sum()
        return f"Total sales: **${total:,}**"
    return "Sorry, I didn’t understand that question."
