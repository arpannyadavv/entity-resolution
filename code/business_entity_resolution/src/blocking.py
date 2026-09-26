"""
Step 2: Blocking / Candidate Generation (v2 - Enhanced)
=========================================================
Multi-strategy blocking to cut search space from O(N*M) to small candidate sets.

Strategies (union per entity):
  1. Word-level TF-IDF cosine similarity  (top-K per entity)
  2. Character 3-gram TF-IDF              (top-K per entity) ← NEW
  3. Country-prefixed name prefix (3 + 4 + 5 chars)
  4. ZIP / PIN code
  5. Soundex / phonetic blocking
  6. First-word exact match
  7. Sorted-token blocking (handles word-order variations) ← NEW
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from preprocess import (
    clean_name, clean_address, get_blocking_key,
    get_name_prefix, extract_zipcode, get_first_token,
    extract_tokens, get_char_ngrams, get_name_chars,
    get_sorted_token_key
)


# ---------------------------------------------------------------------------
# Soundex
# ---------------------------------------------------------------------------
def soundex(name: str) -> str:
    """Simple Soundex encoding."""
    name = re.sub(r'[^a-z]', '', name.lower())
    if not name:
        return '0000'
    codes = {'bfpv': '1', 'cgjkqsxyz': '2', 'dt': '3',
             'l': '4', 'mn': '5', 'r': '6'}
    result = name[0].upper()
    prev_code = ''
    for char in name[1:]:
        code = ''
        for letters, digit in codes.items():
            if char in letters:
                code = digit
                break
        if code and code != prev_code:
            result += code
        prev_code = code
        if len(result) == 4:
            break
    return result.ljust(4, '0')


# ---------------------------------------------------------------------------
# Inverted index helpers
# ---------------------------------------------------------------------------
def build_inverted_index(df: pd.DataFrame, key_fn) -> dict:
    index = defaultdict(list)
    for _, row in df.iterrows():
        key = key_fn(row)
        if key and key != '_':
            index[key].append(row['entity_id'])
    return index


def get_candidates_from_index(s1_row, index: dict, key_fn) -> set:
    key = key_fn(s1_row)
    return set(index.get(key, []))


# ---------------------------------------------------------------------------
# Word-level TF-IDF Blocker
# ---------------------------------------------------------------------------
class TFIDFBlocker:
    """
    Builds a word-level TF-IDF index over business names and finds top-K
    candidates using sparse cosine similarity.
    """

    def __init__(self, top_k: int = 25, ngram_range: tuple = (1, 2)):
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.vocab = {}
        self.idf = {}
        self.entity_ids = []
        self.tfidf_matrix = []

    def _tokenize(self, text: str) -> list:
        tokens = clean_name(text).split()
        result = list(tokens)
        if self.ngram_range[1] >= 2:
            for i in range(len(tokens) - 1):
                result.append(tokens[i] + '_' + tokens[i + 1])
        return result

    def fit(self, df: pd.DataFrame):
        print(f"  Building word TF-IDF index on {len(df):,} records...")
        df_counts = defaultdict(int)
        all_docs = []
        self.entity_ids = list(df['entity_id'])

        for name in tqdm(df['business_name'].fillna(''), desc="  Word-TF-IDF tokenize", ncols=80):
            tokens = self._tokenize(name)
            doc = {}
            for t in tokens:
                doc[t] = doc.get(t, 0) + 1
            all_docs.append(doc)
            for t in set(tokens):
                df_counts[t] += 1

        N = len(df)
        self.vocab = {t: i for i, (t, c) in enumerate(df_counts.items())
                      if 2 <= c <= N * 0.9}
        print(f"  Word vocab size: {len(self.vocab):,}")

        self.idf = {t: math.log((N + 1) / (c + 1)) + 1
                    for t, c in df_counts.items() if t in self.vocab}

        self.tfidf_matrix = []
        for doc in all_docs:
            row = {}
            norm = 0.0
            for t, tf in doc.items():
                if t in self.vocab:
                    val = tf * self.idf[t]
                    row[self.vocab[t]] = val
                    norm += val * val
            norm = math.sqrt(norm) if norm > 0 else 1.0
            self.tfidf_matrix.append({k: v / norm for k, v in row.items()})
        print(f"  Word TF-IDF index built.")

    def query(self, name: str, top_k: int = None) -> list:
        if top_k is None:
            top_k = self.top_k
        tokens = self._tokenize(name)
        query_vec = {}
        for t in tokens:
            if t in self.vocab:
                query_vec[self.vocab[t]] = query_vec.get(self.vocab[t], 0) + 1
        if not query_vec:
            return []
        # Normalize query
        norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if norm > 0:
            query_vec = {k: v / norm for k, v in query_vec.items()}
        # Score docs
        scores = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            score = sum(query_vec.get(k, 0) * v for k, v in doc_vec.items())
            if score > 0.01:
                scores.append((score, self.entity_ids[idx]))
        scores.sort(reverse=True)
        return [eid for _, eid in scores[:top_k]]


# ---------------------------------------------------------------------------
# Character N-gram TF-IDF Blocker (NEW in v2)
# ---------------------------------------------------------------------------
class CharNgramTFIDFBlocker:
    """
    TF-IDF index over CHARACTER n-grams of business names (spaces removed).
    
    This is critical for handling:
      - Abbreviations:  "BUS CTR" ≈ "BUSINESS CENTER" (shared char 3-grams)
      - Typos:          minor character-level edits
      - Word boundaries: "PVTLTD" ≈ "PRIVATE LIMITED"
    """

    def __init__(self, top_k: int = 20, n: int = 3):
        self.top_k = top_k
        self.n = n
        self.vocab = {}
        self.idf = {}
        self.entity_ids = []
        self.tfidf_matrix = []

    def _tokenize(self, name: str) -> list:
        chars = get_name_chars(name)
        return get_char_ngrams(chars, self.n)

    def fit(self, df: pd.DataFrame):
        print(f"  Building char-{self.n}gram TF-IDF index on {len(df):,} records...")
        df_counts = defaultdict(int)
        all_docs = []
        self.entity_ids = list(df['entity_id'])

        for name in tqdm(df['business_name'].fillna(''), desc=f"  Char{self.n}gram tokenize", ncols=80):
            tokens = self._tokenize(name)
            doc = {}
            for t in tokens:
                doc[t] = doc.get(t, 0) + 1
            all_docs.append(doc)
            for t in set(tokens):
                df_counts[t] += 1

        N = len(df)
        # Keep n-grams that appear in at least 2 docs and at most 98% of docs
        self.vocab = {t: i for i, (t, c) in enumerate(df_counts.items())
                      if 2 <= c <= N * 0.98}
        print(f"  Char-{self.n}gram vocab size: {len(self.vocab):,}")

        self.idf = {t: math.log((N + 1) / (c + 1)) + 1
                    for t, c in df_counts.items() if t in self.vocab}

        self.tfidf_matrix = []
        for doc in all_docs:
            row = {}
            norm = 0.0
            for t, tf in doc.items():
                if t in self.vocab:
                    val = tf * self.idf[t]
                    row[self.vocab[t]] = val
                    norm += val * val
            norm = math.sqrt(norm) if norm > 0 else 1.0
            self.tfidf_matrix.append({k: v / norm for k, v in row.items()})
        print(f"  Char-{self.n}gram TF-IDF index built.")

    def query(self, name: str, top_k: int = None) -> list:
        if top_k is None:
            top_k = self.top_k
        tokens = self._tokenize(name)
        query_vec = {}
        for t in tokens:
            if t in self.vocab:
                query_vec[self.vocab[t]] = query_vec.get(self.vocab[t], 0) + 1
        if not query_vec:
            return []
        norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if norm > 0:
            query_vec = {k: v / norm for k, v in query_vec.items()}
        scores = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            score = sum(query_vec.get(k, 0) * v for k, v in doc_vec.items())
            if score > 0.05:   # minimum similarity threshold to keep memory low
                scores.append((score, self.entity_ids[idx]))
        scores.sort(reverse=True)
        return [eid for _, eid in scores[:top_k]]


# ---------------------------------------------------------------------------
# Main Blocker (multi-strategy union)
# ---------------------------------------------------------------------------
class EntityBlocker:
    """
    Multi-strategy blocker that combines many signals to generate a small,
    high-recall candidate set for each S1 entity.
    
    v2 adds:
      - Character n-gram TF-IDF (very powerful for abbreviations)
      - Sorted-token blocking (handles word-order variations)
      - Extended prefix lengths (3, 4, 5 chars)
    """

    def __init__(self, top_k_tfidf: int = 25, top_k_char: int = 20):
        self.top_k_tfidf = top_k_tfidf
        self.top_k_char = top_k_char
        # Word TF-IDF
        self.tfidf_s2 = None
        self.tfidf_s3 = None
        # Char n-gram TF-IDF (new)
        self.char3_s2 = None
        self.char3_s3 = None
        # Inverted indexes
        self.prefix_idx_s2 = {}
        self.prefix_idx_s3 = {}
        self.zip_idx_s2 = {}
        self.zip_idx_s3 = {}
        self.soundex_idx_s2 = {}
        self.soundex_idx_s3 = {}
        self.firstword_idx_s2 = {}
        self.firstword_idx_s3 = {}
        self.sorted_token_idx_s2 = {}   # new
        self.sorted_token_idx_s3 = {}   # new
        self.country_idx_s2 = {}
        self.country_idx_s3 = {}

    def _build_indexes(self, df: pd.DataFrame, source: str):
        prefix_idx = defaultdict(set)
        zip_idx = defaultdict(set)
        soundex_idx = defaultdict(set)
        firstword_idx = defaultdict(set)
        sorted_token_idx = defaultdict(set)
        country_idx = defaultdict(set)

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  Building {source} indexes", ncols=80):
            eid = row['entity_id']
            name = str(row['business_name']) if pd.notna(row['business_name']) else ''
            addr = str(row['business_address']) if pd.notna(row['business_address']) else ''
            country = str(row['country']) if pd.notna(row['country']) else ''

            cc = country.lower().strip()[:2]

            # 1. Prefix blocking: 3, 4, 5 chars (country-prefixed)
            for plen in (3, 4, 5):
                pfx = get_name_prefix(name, plen)
                if pfx:
                    prefix_idx[f"{cc}_{pfx}"].add(eid)

            # 2. ZIP / PIN code
            zipcode = extract_zipcode(addr)
            if zipcode:
                zip_idx[zipcode].add(eid)

            # 3. Soundex of first meaningful word
            fw = get_first_token(name)
            if fw:
                sx = soundex(fw)
                soundex_idx[f"{cc}_{sx}"].add(eid)
                firstword_idx[f"{cc}_{fw}"].add(eid)

            # 4. Sorted-token key (new: word-order invariant)
            stk = get_sorted_token_key(name, country)
            if stk:
                sorted_token_idx[stk].add(eid)

            # 5. Country group
            country_idx[country.lower().strip()].add(eid)

        return prefix_idx, zip_idx, soundex_idx, firstword_idx, sorted_token_idx, country_idx

    def fit(self, s2: pd.DataFrame, s3: pd.DataFrame, use_tfidf: bool = True):
        """Build all indexes from S2 and S3 candidate pools."""
        print("\n[Blocker] Building S2 indexes...")
        (self.prefix_idx_s2, self.zip_idx_s2, self.soundex_idx_s2,
         self.firstword_idx_s2, self.sorted_token_idx_s2,
         self.country_idx_s2) = self._build_indexes(s2, 'S2')

        print("\n[Blocker] Building S3 indexes...")
        (self.prefix_idx_s3, self.zip_idx_s3, self.soundex_idx_s3,
         self.firstword_idx_s3, self.sorted_token_idx_s3,
         self.country_idx_s3) = self._build_indexes(s3, 'S3')

        if use_tfidf:
            print("\n[Blocker] Building word TF-IDF for S2...")
            self.tfidf_s2 = TFIDFBlocker(top_k=self.top_k_tfidf)
            self.tfidf_s2.fit(s2)
            print("\n[Blocker] Building word TF-IDF for S3...")
            self.tfidf_s3 = TFIDFBlocker(top_k=self.top_k_tfidf)
            self.tfidf_s3.fit(s3)

            print("\n[Blocker] Building char-3gram TF-IDF for S2...")
            self.char3_s2 = CharNgramTFIDFBlocker(top_k=self.top_k_char, n=3)
            self.char3_s2.fit(s2)
            print("\n[Blocker] Building char-3gram TF-IDF for S3...")
            self.char3_s3 = CharNgramTFIDFBlocker(top_k=self.top_k_char, n=3)
            self.char3_s3.fit(s3)

    def get_candidates(self, s1_row) -> set:
        """Get all candidate IDs for a single S1 entity (union of all strategies)."""
        name = str(s1_row['business_name']) if pd.notna(s1_row['business_name']) else ''
        addr = str(s1_row['business_address']) if pd.notna(s1_row['business_address']) else ''
        country = str(s1_row['country']) if pd.notna(s1_row['country']) else ''
        cc = country.lower().strip()[:2]

        candidates = set()

        # --- Prefix blocking (3, 4, 5 chars) ---
        for plen in (3, 4, 5):
            pfx = get_name_prefix(name, plen)
            key = f"{cc}_{pfx}"
            candidates |= self.prefix_idx_s2.get(key, set())
            candidates |= self.prefix_idx_s3.get(key, set())

        # --- ZIP / PIN blocking ---
        zipcode = extract_zipcode(addr)
        if zipcode:
            candidates |= self.zip_idx_s2.get(zipcode, set())
            candidates |= self.zip_idx_s3.get(zipcode, set())

        # --- Soundex + first-word exact ---
        fw = get_first_token(name)
        if fw:
            sx = soundex(fw)
            candidates |= self.soundex_idx_s2.get(f"{cc}_{sx}", set())
            candidates |= self.soundex_idx_s3.get(f"{cc}_{sx}", set())
            candidates |= self.firstword_idx_s2.get(f"{cc}_{fw}", set())
            candidates |= self.firstword_idx_s3.get(f"{cc}_{fw}", set())

        # --- Sorted-token blocking (new) ---
        stk = get_sorted_token_key(name, country)
        if stk:
            candidates |= self.sorted_token_idx_s2.get(stk, set())
            candidates |= self.sorted_token_idx_s3.get(stk, set())

        # --- Word TF-IDF retrieval ---
        if self.tfidf_s2:
            candidates |= set(self.tfidf_s2.query(name, top_k=self.top_k_tfidf))
        if self.tfidf_s3:
            candidates |= set(self.tfidf_s3.query(name, top_k=self.top_k_tfidf))

        # --- Char n-gram TF-IDF retrieval (new) ---
        if self.char3_s2:
            candidates |= set(self.char3_s2.query(name, top_k=self.top_k_char))
        if self.char3_s3:
            candidates |= set(self.char3_s3.query(name, top_k=self.top_k_char))

        return candidates

    def generate_candidates(self, s1: pd.DataFrame) -> dict:
        """Generate candidate dict {s1_entity_id -> set of candidate IDs}."""
        print(f"\n[Blocker] Generating candidates for {len(s1):,} S1 entities...")
        results = {}
        for _, row in tqdm(s1.iterrows(), total=len(s1), desc="  Blocking", ncols=80):
            eid = row['entity_id']
            results[eid] = self.get_candidates(row)

        total_cands = sum(len(v) for v in results.values())
        print(f"  Total candidates: {total_cands:,}")
        print(f"  Avg per S1 entity: {total_cands / len(s1):.1f}")
        return results


if __name__ == '__main__':
    DTYPES = {'entity_id': str, 'business_name': str, 'business_address': str, 'country': str}
    print("Loading samples for blocking test...")
    s2 = pd.read_csv('../../../dataset/train/train_source2.tsv', sep='\t', dtype=DTYPES, nrows=10000)
    s3 = pd.read_csv('../../../dataset/train/train_source3.tsv', sep='\t', dtype=DTYPES, nrows=10000)
    s1 = pd.read_csv('../../../dataset/train/train_source1.tsv', sep='\t', dtype=DTYPES, nrows=100)

    blocker = EntityBlocker(top_k_tfidf=25, top_k_char=20)
    blocker.fit(s2, s3, use_tfidf=True)
    candidates = blocker.generate_candidates(s1)
    eid = list(candidates.keys())[0]
    print(f"\nSample: {eid} → {len(candidates[eid])} candidates")
    print(f"Sample IDs: {list(candidates[eid])[:5]}")
