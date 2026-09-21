import argparse
import re
import pandas as pd
import spacy
import emoji
import contractions

from tqdm import tqdm

tqdm.pandas()

NEGATION_WORDS = {"not", "no", "nor", "never"}
CUSTOM_STOPWORDS = ["lol", "lmao", "omg", "yeah", "gonna", "wanna", "gotta", "u", "ur",
                    "user", "go", "man"]

SLANG_DICT = {
    "u": "you", "ur": "your", "r": "are", "y": "why",
    "im": "i am", "dont": "do not", "cant": "cannot", "wont": "will not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not", "werent": "were not",
    "doesnt": "does not", "didnt": "did not", "shouldnt": "should not",
    "wouldnt": "would not", "couldnt": "could not", "havent": "have not",
    "hasnt": "has not",
}

MODERN_SLANG = {
    "no cap": "no lie", "sus": "suspicious", "slay": "amazing",
    "rizz": "charisma", "ghosted": "ignored", "salty": "bitter",
    "stan": "adore", "cringe": "embarrassing", "goat": "best",
    "based": "agree", "cappin": "lying", "bet": "okay",
    "bussin": "delicious", "fr": "for real", "iykyk": "you know",
    "ngl": "honestly", "smh": "disappointed", "deadass": "seriously",
}

def normaliza_slang(text: str) -> str:
    text = contractions.fix(text, slang=True)
    words = text.split()
    words = [SLANG_DICT.get(word, word) for word in words]
    text = " ".join(words)
    for slang, formal in MODERN_SLANG.items():
        text = re.sub(rf"\b{slang}\b", formal, text, flags=re.IGNORECASE)
    return text

def load_nlp_spacy():
    nlp = spacy.load("en_core_web_sm")
    for word in CUSTOM_STOPWORDS:
        nlp.vocab[word].is_stop = True
    return nlp

def reduce_text(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)

def clean_for_model(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = contractions.fix(text, slang=True)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    # Biarkan emoji, kapitalisasi, tanda baca tetap utuh untuk model
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " http ", text)
    text = re.sub(r"@\w+", " @user ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = reduce_text(text)
    text = normaliza_slang(text)
    text = emoji.replace_emoji(text, replace="")
    # text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_visualization(text: str, nlp) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    lemmaz = []
    negate = False
    for token in doc:
        if token.lower_ in NEGATION_WORDS:
            negate = True
            continue
        if negate and not token.is_punct:
            lemma = f"NOT_{token.lemma_}"
            negate = False
        else:
            lemma = token.lemma_
        if not token.is_stop and not nlp.vocab[lemma].is_stop and not token.is_punct and len(lemma.strip()) > 1:
            lemmaz.append(lemma)
    return " ".join(lemmaz)

def main(input_path: str, output_path: str, column: str):
    df = pd.read_csv(input_path)

    df['text_model'] = df[column].progress_apply(clean_for_model)

    nlp = load_nlp_spacy()
    df['text_clean'] = df[column].progress_apply(
        lambda text: clean_visualization(clean_text(text), nlp)
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
