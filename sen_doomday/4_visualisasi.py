import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

sns.set_style('whitegrid')
sentiment_color = {"Positif" : "#51895b", "Neutral" : "#c2c8cb", "Negatif" : "#e8706e"}

def count_plot(df: pd.DataFrame, out_directory: str):
    order = df['sentiment'].value_counts().index.tolist()
    colors = [sentiment_color.get(s, "#3498db") for s in order]

    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='sentiment', order=order, hue='sentiment', hue_order=order, palette=colors, legend=False)
    ax.set_title("Countplot Sentimen Komentar YouTube", fontsize=12, fontweight="bold")
    ax.set_xlabel("Kategori Sentimen", fontsize=10)
    ax.set_ylabel("Jumlah Komentar", fontsize=10)

    total_data = len(df['sentiment'])
    for p in ax.patches:
        count = p.get_height()
        percentage = (count / total_data) * 100
        label = f'{count} ({percentage:.1f}%)'
        x = p.get_x() + p.get_width() / 2
        y = count + (total_data * 0.01)
        ax.annotate(label, (x, y), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(out_directory, "1_countplot_sentiment.png")
    plt.savefig(path, dpi=250)
    plt.close()

def pie_plot(df: pd.DataFrame, out_directory: str):
    counts = df['sentiment'].value_counts()
    colors = [sentiment_color.get(s, "#3498db") for s in counts.index]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    plt.title("Persentase Sentimen Komentar YouTube", fontsize=12, fontweight="bold")
    plt.legend(counts.index, title="Sentimen", loc="best")
    plt.axis("equal")
    plt.tight_layout()
    path = os.path.join(out_directory, "2_piechart_sentiment.png")
    plt.savefig(path, dpi=250)
    plt.close()

def word_cloud_plot(df: pd.DataFrame, out_directory: str, column_clean: str):
    text = " ".join(df[column_clean].dropna().astype(str))
    word_cloud = WordCloud(width=1500, height=600, background_color='white', colormap='viridis').generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud Komentar YouTube", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_directory, "3_wordcloud_sentiment.png")
    plt.savefig(path, dpi=250)
    plt.close()

def top_plot(df: pd.DataFrame, out_directory: str, column_clean: str, top_n: int = 5):
    word = " ".join(df[column_clean].dropna().astype(str)).split()
    counter = Counter(word)
    top_word = counter.most_common(top_n)
    words, freq = zip(*top_word)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(freq), y=list(words), hue=list(words), palette="Blues_r", legend=False)
    ax.set_title(f"Top {top_n} Kata yang Paling Sering Muncul", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frekuensi Kemunculan", fontsize=10)
    ax.set_ylabel("Kata", fontsize=10)

    for index, value in enumerate(freq):
        ax.text(value, index, f" {value}", va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_directory, "4_top_words_sentiment.png")
    plt.savefig(path, dpi=250)
    plt.close()
    return top_word

def main(input_path: str, output_path: str, column: str, top_n: int):
    os.makedirs(output_path, exist_ok=True)
    df = pd.read_csv(input_path)

    count_plot(df, output_path)
    pie_plot(df, output_path)
    word_cloud_plot(df, output_path, column)
    top_plot(df, output_path, column, top_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisasi Hasil Analisis Sentimen")
    parser.add_argument("--in", dest="input_path", default="Komentar_Youtube_Model.csv")
    parser.add_argument("--out", dest="output_path", default="Komentar_Youtube_Visualization")
    parser.add_argument("--col", dest="column", default="text_clean")
    parser.add_argument("--top-n", dest="top_n", type=int, default=10)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.column, args.top_n)