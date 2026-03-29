from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = "cyeye-genz-dataset-analyzer"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", transparent=False)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def build_bar_chart(series: pd.Series, title: str, top_n: int = 12) -> str | None:
    s = series.dropna().astype(str).value_counts().head(top_n)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=35)
    return fig_to_base64(fig)


def build_hist_chart(series: pd.Series, title: str) -> str | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(s, bins=min(30, max(8, int(len(s) ** 0.5))))
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return fig_to_base64(fig)


def analyze_dataframe(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

    columns = []
    for col in df.columns:
        s = df[col]
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "sample_values": [str(x)[:120] for x in s.dropna().astype(str).head(3).tolist()],
        }
        if pd.api.types.is_numeric_dtype(s):
            clean = pd.to_numeric(s, errors="coerce")
            info.update({
                "min": None if clean.dropna().empty else round(float(clean.min()), 4),
                "max": None if clean.dropna().empty else round(float(clean.max()), 4),
                "mean": None if clean.dropna().empty else round(float(clean.mean()), 4),
            })
        else:
            vc = s.dropna().astype(str).value_counts().head(5)
            info["top_values"] = [{"value": str(k)[:80], "count": int(v)} for k, v in vc.items()]
        columns.append(info)

    preview = df.head(20).fillna("").astype(str).to_dict(orient="records")

    charts = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    if numeric_cols:
        for col in numeric_cols[:3]:
            img = build_hist_chart(df[col], f"Distribution • {col}")
            if img:
                charts.append({"title": f"Distribution • {col}", "image": img})
    if categorical_cols:
        for col in categorical_cols[:3]:
            img = build_bar_chart(df[col], f"Top values • {col}")
            if img:
                charts.append({"title": f"Top values • {col}", "image": img})

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(12)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Missing values by column")
        ax.set_xlabel("")
        ax.set_ylabel("Missing count")
        ax.tick_params(axis="x", labelrotation=35)
        charts.append({"title": "Missing values by column", "image": fig_to_base64(fig)})

    return {
        "summary": summary,
        "dtype_counts": dtype_counts,
        "columns": columns,
        "preview": preview,
        "preview_columns": [str(c) for c in df.columns],
        "charts": charts,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    analysis = None
    filename = None

    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Pick a CSV file first.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Only CSV files are supported in this version.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        try:
            df = pd.read_csv(save_path)
        except UnicodeDecodeError:
            df = pd.read_csv(save_path, encoding="latin-1")
        except Exception as exc:
            flash(f"Could not read CSV: {exc}")
            return redirect(url_for("index"))

        analysis = analyze_dataframe(df)

    return render_template("index.html", analysis=analysis, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
