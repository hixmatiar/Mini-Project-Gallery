import pandas as pd
import argparse
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess_tweet(text: str) -> str:
    tokens = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1:
            t = "@user"
        elif t.startswith("http"):
            t = "http"
        tokens.append(t)
    return " ".join(tokens)


def main(input_path: str, output_path: str, column: str, batch_size: int):
    df = pd.read_csv(input_path)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    dev = device()
    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=dev,
        truncation=True,
        max_length=512,
    )

    texts = df[column].fillna("").astype(str).tolist()

    labels, scores = [], []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch = [preprocess_tweet(t) for t in batch]
        result = pipe(batch, batch_size=batch_size)
        for r in result:
            labels.append(r["label"])
            scores.append(r["score"])

    df["sentiment"] = labels
    df["sentiment_score"] = scores

    df.to_csv(output_path, index=False)
    print(f"Hasil Disimpan ke {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis Sentimen Menggunakan RoBERTa")
    parser.add_argument("--in", dest="input_path", default="Komentar_Youtube_Clean.csv")
    parser.add_argument("--out", dest="output_path", default="Komentar_Youtube_Model.csv")
    parser.add_argument("--col", dest="column", default="text_model")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.column, args.batch_size)