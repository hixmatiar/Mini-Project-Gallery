import pandas as pd
import argparse
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "mdhugol/indonesia-bert-sentiment-classification"

label_map = {
    "LABEL_0" : "Positif",
    "LABEL_1" : "Neutral",
    "LABEL_2" : "Negatif"
}

def device():
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return "cpu"
    
def label_mapping(model):
    id2label = model.config.id2label
    mapped = {}
    for index, label in id2label.items():
        label_upper = str(label).upper()
        if "POS" in label_upper:
            mapped[label] = "Positif"
        elif "NEU" in label_upper or "NET" in label_upper:
            mapped[label] = "Netral"
        elif "NEG" in label_upper:
            mapped[label] = "Negatif"
        else:
            mapped[label] = label_map.get(label, label)
    return mapped

def main(input_path: str, output_path: str, column: str, batch_size: int):
    df = pd.read_csv(input_path)

    print("Install IndoBERT model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    map_label = label_mapping(model)

    devices = device()
    pipeline_sentiment = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if devices == 'mps' else -1,
        truncation=True,
        max_length=512
    ) 

    texts = df[column].fillna("").astype(str).tolist()

    labels, scores = [], []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        result = pipeline_sentiment(batch, batch_size=batch_size)
        for re in result:
            labels.append(map_label.get(re['label'], re['label']))
            scores.append(re['score'])
    

    df['sentiment'] = labels
    df['sentiment_score'] = scores

    df.to_csv(output_path, index=False)
    print(f"Hasil Disimpan ke {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis Sentimen Menggunakan indoBERT")
    parser.add_argument("--in", dest="input_path", default="Komentar_Youtube_Clean.csv")
    parser.add_argument("--out", dest="output_path", default="Komentar_Youtube_Model.csv")
    parser.add_argument("--col", dest="column", default="text_model")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.column, args.batch_size)