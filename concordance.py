# concordance.py

import stanza
import math
import pandas as pd
from collections import Counter, defaultdict


# Load model once
nlp = stanza.Pipeline('hy')

with open('stopwords-hy.txt', encoding='utf8') as f:
    armenian_stopwords = {x.strip() for x in f}


def preprocess(text, on_progress=None):
    # Step 1: Split into sentences using just the tokenizer
    nlp_tokenize = stanza.Pipeline(lang='hy', processors='tokenize', use_gpu=False)
    doc = nlp_tokenize(text)
    sentences_raw = [sentence.text for sentence in doc.sentences]

    # Step 2: Lemmatize one sentence at a time
    nlp_morph = stanza.Pipeline(lang='hy', processors='tokenize,pos,lemma', use_gpu=False)

    sentences_lemmas = []
    total = len(sentences_raw)

    for idx, sent in enumerate(sentences_raw):
        sent_doc = nlp_morph(sent)
        sentence_lemmas = []

        for s in sent_doc.sentences:
            for word in s.words:
                sentence_lemmas.append({'text': word.text, 'lemma': word.lemma})

        sentences_lemmas.append(sentence_lemmas)

        # Update progress
        if on_progress:
            on_progress(idx + 1, total)

    return sentences_lemmas

def search_concordance_with_lemma(sentences_lemmas, keyword_lemma, window=3):
    results = []
    keyword_lemma = keyword_lemma.lower()

    for sent in sentences_lemmas:
        lemmas = [t['lemma'].lower() for t in sent]
        for i, lemma in enumerate(lemmas):
            if lemma == keyword_lemma:
                left = [t['text'] for t in sent[max(0, i-window):i]]
                word = sent[i]['text']
                right = [t['text'] for t in sent[i+1:i+1+window]]
                results.append((' '.join(left), word, ' '.join(right)))
    return results

def format_concordance(results, window=3):
    formatted = []
    for left, word, right in results:
        left_parts = left.split()
        right_parts = right.split()

        left_text = '...' + left if len(left_parts) == window else left
        right_text = right + '...' if len(right_parts) == window else right

        # center = f'***{word}***'
        line = f"{left_text} <span style='color:#fff; font-weight:bold;'>{word}</span> {right_text}"
        formatted.append(line)
    return formatted

def get_word_frequencies(text, use_lemmas=True, remove_stopwords=True):
    doc = nlp(text)
    words = []

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.isalpha():
                lemma_or_text = word.lemma if use_lemmas else word.text
                token = lemma_or_text.lower()
                if remove_stopwords and token in armenian_stopwords:
                    continue
                words.append(token)

    return Counter(words)

def find_collocations(text, keyword, window=5, measure="PMI", use_lemmas=True, ordering = "Score"):
    doc = nlp(text)

    # Use lemmas or raw text based on flag
    tokens = [
        (word.lemma.lower() if use_lemmas else word.text.lower())
        for sent in doc.sentences for word in sent.words
        if word.text.isalpha()
    ]

    total_tokens = len(tokens)
    keyword = keyword.lower()
    keyword_count = tokens.count(keyword)

    if keyword_count == 0:
        return pd.DataFrame(columns=["Collocate", "Frequency", "Frequency(L)", "Frequency(R)", "Score"])

    colloc_freq = defaultdict(int)
    colloc_left = defaultdict(int)
    colloc_right = defaultdict(int)
    word_freq = Counter(tokens)

    for i, token in enumerate(tokens):
        if token == keyword:
            for j in range(1, window + 1):
                if i - j >= 0:
                    word = tokens[i - j]
                    colloc_freq[word] += 1
                    colloc_left[word] += 1
                if i + j < total_tokens:
                    word = tokens[i + j]
                    colloc_freq[word] += 1
                    colloc_right[word] += 1

    results = []
    for word in colloc_freq:
        f_xy = colloc_freq[word]
        f_x = word_freq[word]
        f_y = keyword_count

        if measure == "PMI":
            p_xy = f_xy / (total_tokens - 2 * window)
            p_x = f_x / total_tokens
            p_y = f_y / total_tokens
            score = math.log2(p_xy / (p_x * p_y)) if p_x * p_y > 0 else 0
        elif measure == "T-Score":
            expected = (f_x * f_y) / total_tokens
            score = (f_xy - expected) / math.sqrt(f_xy) if f_xy > 0 else 0
        else:
            score = 0

        results.append({
            "Collocate": word,
            "Frequency": f_xy,
            "Frequency(L)": colloc_left[word],
            "Frequency(R)": colloc_right[word],
            "Score": round(score, 3)
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by=ordering, ascending=False).reset_index(drop=True)
    return df

def get_clusters(text, use_lemmas=True, remove_stopwords=True, ngram_size=2):
    doc = nlp(text)

    # Build list of processed sentences
    sentences = []
    for sentence in doc.sentences:
        tokens = []
        for word in sentence.words:
            token = word.lemma.lower() if use_lemmas else word.text.lower()
            if token.isalpha():  # only keep alphabetic
                if remove_stopwords and token in armenian_stopwords:
                    continue
                tokens.append(token)
        sentences.append(tokens)

    # Count n-grams and their sentence ranges
    ngram_counter = Counter()
    ngram_sentences = {}

    for idx, sent in enumerate(sentences):
        seen_ngrams = set()
        for i in range(len(sent) - ngram_size + 1):
            ngram = " ".join(sent[i:i + ngram_size])
            ngram_counter[ngram] += 1
            if ngram not in seen_ngrams:
                ngram_sentences.setdefault(ngram, set()).add(idx)
                seen_ngrams.add(ngram)

    # Build final dictionary
    result = {
        ngram: {
            "frequency": freq,
            "range": len(ngram_sentences[ngram])
        } for ngram, freq in ngram_counter.items()
    }

    return result
