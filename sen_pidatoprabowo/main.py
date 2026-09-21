# import library
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from wordcloud import WordCloud
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline

sentiment_model = 'bert'
# https://www.youtube.com/watch?v=s-LnbQYNkwA&t=8s
# scrapping data from youtube
def download_comments(video_url, limit=200):
    downloader = YoutubeCommentDownloader()
    comments = []
    for comment in tqdm(downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR),
                         desc="Mengunduh komentar"):
        comments.append({
            "author": comment["author"],
            "text": comment["text"],
            "likes": comment["votes"],
            "time": comment["time"]
        })
        if len(comments) >= limit:
            break
 
    df = pd.DataFrame(comments)
    print(f"Berhasil mengunduh {len(df)} komentar.")
    return df

# cleaning data
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  
    text = re.sub(r"@\w+", "", text)     
    text = re.sub(r"[^\w\s]", " ", text)       
    text = re.sub(r"\d+", "", text)            
    text = text.lower()                           
    text = re.sub(r"\s+", " ", text).strip()           
    return text

# data preprocessing
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def preprocess_text(text, use_stemming=False):
    text = stopword_remover.remove(text)
    if use_stemming:
        text = stemmer.stem(text)
    return text

# analysis sentiment use indoBERT
def load_sentiment_model():
    import torch
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    device = 0 if torch.backends.mps.is_available() else -1
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device
    )
    return sentiment_pipeline

def analyze_sentiment(df, sentiment_pipeline):
    label_map = {
        "LABEL_0": "positive",
        "LABEL_1": "neutral",
        "LABEL_2": "negative"
    }
    sentiments = []
    scores = []
    for text in tqdm(df["clean_text"], desc="Menganalisis sentimen"):
        if len(text.strip()) == 0:
            sentiments.append("neutral")
            scores.append(0.0)
            continue
        # Batasi panjang teks ke 512 token (batas maksimal BERT)
        result = sentiment_pipeline(text[:512])[0]
        sentiments.append(label_map.get(result["label"], result["label"]))
        scores.append(result["score"])
    df["sentiment"] = sentiments
    df["confidence"] = scores
    return df

# data visualization
def visualize_results(df):
    """
    Membuat 2 visualisasi:
    1. Bar chart distribusi jumlah sentimen (positif/netral/negatif),
       lengkap dengan label jumlah komentar DAN persentasenya di atas tiap bar
    2. Wordcloud dari keseluruhan komentar
    """
    sns.set_style("whitegrid")
 
    # --- Hitung jumlah & persentase per kategori sentimen ---
    order = ["positive", "neutral", "negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    total = counts.sum()
    percentages = (counts / total * 100).round(1)
 
    # --- Bar chart distribusi sentimen ---
    plt.figure(figsize=(6, 5))
    palette = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#F44336"}
    colors = [palette[cat] for cat in order]
 
    bars = plt.bar(order, counts.values, color=colors)
 
    # Tambahkan label "jumlah (persentase%)" di atas tiap bar
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,   # posisi x: tengah bar
            bar.get_height() + (total * 0.01),    # posisi y: sedikit di atas bar
            f"{count}\n({pct}%)",                  # teks: jumlah baris pertama, persentase baris kedua
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
 
    plt.title("Distribusi Sentimen Komentar YouTube")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah Komentar")
    plt.ylim(0, counts.max() * 1.2)  # kasih ruang ekstra di atas supaya label tidak terpotong
    plt.tight_layout()
    plt.savefig("distribusi_sentimen.png", dpi=150)
    plt.close()  # tutup otomatis, tidak menunggu jendela ditutup manual (penting saat run via terminal)
    print("Grafik distribusi sentimen disimpan: distribusi_sentimen.png")
 
    # --- Pie chart persentase sentimen ---
    plt.figure(figsize=(6, 6))
    # Hanya tampilkan kategori yang jumlahnya > 0 (biar tidak ada label 0%)
    mask = counts.values > 0
    plt.pie(
        counts.values[mask],
        labels=[cat.capitalize() for cat, m in zip(order, mask) if m],
        colors=[colors[i] for i, m in enumerate(mask) if m],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*total))})",  # tampilkan % dan jumlah
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"}
    )
    plt.title("Persentase Sentimen Komentar YouTube")
    plt.tight_layout()
    plt.savefig("persentase_sentimen.png", dpi=150)
    plt.close()
    print("Pie chart persentase sentimen disimpan: persentase_sentimen.png")
 
    # --- Wordcloud ---
    all_text = " ".join(df["clean_text"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud Komentar")
    plt.tight_layout()
    plt.savefig("wordcloud_komentar.png", dpi=150)
    plt.close()
    print("Wordcloud disimpan: wordcloud_komentar.png")

# main function
if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=s-LnbQYNkwA&t=8s"
    JUMLAH_KOMENTAR = 1000 
    df = download_comments(VIDEO_URL, limit=JUMLAH_KOMENTAR)

    df["clean_text"] = df["text"].apply(clean_text)

    df["clean_text"] = df["clean_text"].apply(preprocess_text)

    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    sentiment_pipeline = load_sentiment_model()
    df = analyze_sentiment(df, sentiment_pipeline)

    print("\n=== RINGKASAN HASIL ===")
    print(df["sentiment"].value_counts())
    print(f"\nPersentase:\n{df['sentiment'].value_counts(normalize=True) * 100}")
    visualize_results(df)

    df.to_csv("sentimen_pidato-prabowo-10-07-2026.csv", index=False, encoding="utf-8-sig")
    print("\nHasil disimpan ke 'sentimen_pidato-prabowo-10-07-2026.csv'")