import argparse
import re
import pandas as pd
import saka

from tqdm import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary

tqdm.pandas()

def reduce_text(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = reduce_text(text)
    text = saka.normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text

# stemmer = StemmerFactory().create_stemmer()
# stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# def stemmer_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     return stemmer.stem(text)

def lemmatization(text: str) -> str:
    words = text.split()
    lemmas = []
    for word in words:
        try:
            root_word = saka.analyze(word)["root"]
            lemmas.append(root_word if root_word else word)
        except:
            lemmas.append(word)
    return " ".join(lemmas)

word_stopword_custom = ['dak', 'pra', 'pak', 'di', 'iya', 'yang', ]

def remove_stopwords():
    # if not isinstance(text, str):
    #     return ""
    sastrawi_stopword = StopWordRemoverFactory().get_stop_words()
    saka_stopword = saka.get_stopwords("id")
    combine_stopwords = (set(sastrawi_stopword) | set(saka_stopword) | set(word_stopword_custom))
    dictionary_stopword = ArrayDictionary((list(combine_stopwords)))
    return StopWordRemover(dictionary_stopword)

re_stopword = remove_stopwords()

def clean_visualization(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = lemmatization(text)
    text = re_stopword.remove(text)
    return text

def main(input_path: str, output_path: str, column: str):
    df = pd.read_csv(input_path)
    
    df['text_model'] = df[column].progress_apply(
        lambda text: clean_text(text)
    )

    df['text_clean'] = df['text_model'].progress_apply(
        lambda text: clean_visualization(text)
    )

    df.to_csv(output_path, index=False)
    print("file hasil cleaning berhasil disimpan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleaning Komentar Youtube')
    parser.add_argument("--in", dest="input_path", default="Komentar_Youtube.csv")
    parser.add_argument("--out", dest="output_path", default="Komentar_Youtube_Clean.csv")
    parser.add_argument("--col", dest="column", default="text")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.column)