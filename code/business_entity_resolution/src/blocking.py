"""
Step 2: Blocking / Candidate Generation
Multi-strategy blocking to cut search space from O(N*M) to small candidate sets.

Strategies used (applied per country group):
  1. TF-IDF cosine similarity on business names  (top-K per entity)
  2. Name prefix blocking (first 3-4 chars)
  3. ZIP/PIN code blocking
  4. Sorted-token first-word blocking
  5. Soundex/phonetic blocking for English names

Union of all strategies → candidate_pairs.tsv
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
    extract_tokens
)


# ---- Soundex ----
def soundex(name: str) -> str:
    """Simple Soundex encoding for phonetic blocking."""
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


def build_inverted_index(df: pd.DataFrame, key_fn) -> dict:
    """Build inverted index: key -> list of entity_ids."""
    index = defaultdict(list)
    for _, row in df.iterrows():
        key = key_fn(row)
        if key and key != '_':
            index[key].append(row['entity_id'])
    return index


def get_candidates_from_index(s1_row, index: dict, key_fn) -> set:
    """Get candidate entity_ids for an S1 row from an inverted index."""
    key = key_fn(s1_row)
    return set(index.get(key, []))


# ---- TF-IDF based blocking (vectorized, memory-efficient) ----
class TFIDFBlocker:
    """
    Builds a TF-IDF index over business names and finds top-K most similar
    candidates for each query using sparse matrix multiplication.
    """

    def __init__(self, top_k: int = 10, ngram_range: tuple = (1, 2)):
        self.top_k = top_k
        self.ngram_range = ngram_range
        self.vocab = {}
        self.idf = {}
        self.entity_ids = []
        self.tfidf_matrix = None  # shape: (n_candidates, vocab_size) as list of dicts

    def _tokenize(self, text: str) -> list:
        tokens = clean_name(text).split()
        result = list(tokens)
        if self.ngram_range[1] >= 2:
            for i in range(len(tokens) - 1):
                result.append(tokens[i] + '_' + tokens[i + 1])
        return result

    def fit(self, df: pd.DataFrame):
        """Build TF-IDF index from dataframe."""
        print(f"  Building TF-IDF index on {len(df):,} records...")
        # Count document frequencies
        df_counts = defaultdict(int)
        all_docs = []
        self.entity_ids = list(df['entity_id'])
        
        for name in tqdm(df['business_name'].fillna(''), desc="  Tokenizing", ncols=80):
            tokens = self._tokenize(name)
            doc = {}
            for t in tokens:
                doc[t] = doc.get(t, 0) + 1
            all_docs.append(doc)
            for t in set(tokens):
                df_counts[t] += 1
        
        # Build vocab
        N = len(df)
        # Filter very rare and very common tokens
        self.vocab = {t: i for i, (t, c) in enumerate(df_counts.items())
                      if 2 <= c <= N * 0.9}
        vocab_size = len(self.vocab)
        print(f"  Vocab size: {vocab_size:,} terms")
        
        # Compute IDF
        self.idf = {t: math.log((N + 1) / (c + 1)) + 1
                    for t, c in df_counts.items() if t in self.vocab}
        
        # Build TF-IDF sparse rows (as list of dicts)
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
        print(f"  TF-IDF index built.")

    def query(self, name: str, top_k: int = None) -> list:
        """Find top-K most similar entity_ids to query name."""
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
        norm = math.sqrt(sum(v * v * self.idf.get(t, 1) ** 2
                             for t, v in zip(tokens, [1] * len(tokens))
                             if t in self.vocab))
        if norm == 0:
            return []
        
        # Score each document (dot product with sparse rows)
        scores = []
        for idx, doc_vec in enumerate(self.tfidf_matrix):
            score = sum(query_vec.get(k, 0) * v for k, v in doc_vec.items())
            if score > 0:
                scores.append((score, self.entity_ids[idx]))
        
        scores.sort(reverse=True)
        return [eid for _, eid in scores[:top_k]]


# ---- Main Blocker ----
class EntityBlocker:
    """
    Multi-strategy blocker that combines multiple signals to generate
    a small, high-recall candidate set for each S1 entity.
    """

    def __init__(self, top_k_tfidf: int = 15):
        self.top_k_tfidf = top_k_tfidf
        self.tfidf_s2 = None
        self.tfidf_s3 = None
        # Inverted indexes
        self.prefix_idx_s2 = {}
        self.prefix_idx_s3 = {}
        self.zip_idx_s2 = {}
        self.zip_idx_s3 = {}
        self.soundex_idx_s2 = {}
        self.soundex_idx_s3 = {}
        self.firstword_idx_s2 = {}
        self.firstword_idx_s3 = {}
        self.country_idx_s2 = {}
        self.country_idx_s3 = {}

    def _build_indexes(self, df: pd.DataFrame, source: str):
        """Build all inverted indexes for a source dataframe."""
        prefix_idx = defaultdict(set)
        zip_idx = defaultdict(set)
        soundex_idx = defaultdict(set)
        firstword_idx = defaultdict(set)
        country_idx = defaultdict(set)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Building {source} indexes", ncols=80):
            eid = row['entity_id']
            name = str(row['business_name']) if pd.notna(row['business_name']) else ''
            addr = str(row['business_address']) if pd.notna(row['business_address']) else ''
            country = str(row['country']) if pd.notna(row['country']) else ''
            
            cleaned_name = clean_name(name)
            cleaned_country = country.lower().strip()

            # 1. Country-prefixed name prefix (4 chars)
            pfx = get_name_prefix(name, 4)
            if pfx:
                prefix_idx[f"{cleaned_country[:2]}_{pfx}"].add(eid)
            
            # 2. Country-prefixed name prefix (3 chars)
            pfx3 = get_name_prefix(name, 3)
            if pfx3:
                prefix_idx[f"{cleaned_country[:2]}_{pfx3}"].add(eid)

            # 3. ZIP/PIN code
            zipcode = extract_zipcode(addr)
            if zipcode:
                zip_idx[zipcode].add(eid)

            # 4. Soundex of first meaningful word
            fw = get_first_token(name)
            if fw:
                sx = soundex(fw)
                soundex_idx[f"{cleaned_country[:2]}_{sx}"].add(eid)
                firstword_idx[f"{cleaned_country[:2]}_{fw}"].add(eid)

            # 5. Country group
            country_idx[cleaned_country].add(eid)

        return prefix_idx, zip_idx, soundex_idx, firstword_idx, country_idx

    def fit(self, s2: pd.DataFrame, s3: pd.DataFrame, use_tfidf: bool = True):
        """Build all indexes from S2 and S3 candidate pools."""
        print("\n[Blocker] Building S2 indexes...")
        (self.prefix_idx_s2, self.zip_idx_s2,
         self.soundex_idx_s2, self.firstword_idx_s2,
         self.country_idx_s2) = self._build_indexes(s2, 'S2')

        print("\n[Blocker] Building S3 indexes...")
        (self.prefix_idx_s3, self.zip_idx_s3,
         self.soundex_idx_s3, self.firstword_idx_s3,
         self.country_idx_s3) = self._build_indexes(s3, 'S3')

        if use_tfidf:
            print("\n[Blocker] Building TF-IDF index for S2...")
            self.tfidf_s2 = TFIDFBlocker(top_k=self.top_k_tfidf)
            self.tfidf_s2.fit(s2)

            print("\n[Blocker] Building TF-IDF index for S3...")
            self.tfidf_s3 = TFIDFBlocker(top_k=self.top_k_tfidf)
            self.tfidf_s3.fit(s3)

    def get_candidates(self, s1_row) -> set:
        """Get all candidate IDs for a single S1 entity."""
        name = str(s1_row['business_name']) if pd.notna(s1_row['business_name']) else ''
        addr = str(s1_row['business_address']) if pd.notna(s1_row['business_address']) else ''
        country = str(s1_row['country']) if pd.notna(s1_row['country']) else ''
        cleaned_country = country.lower().strip()
        cc = cleaned_country[:2]

        candidates = set()

        # 1. Prefix blocking (4-char and 3-char)
        pfx4 = get_name_prefix(name, 4)
        pfx3 = get_name_prefix(name, 3)
        key4 = f"{cc}_{pfx4}"
        key3 = f"{cc}_{pfx3}"
        candidates |= self.prefix_idx_s2.get(key4, set())
        candidates |= self.prefix_idx_s2.get(key3, set())
        candidates |= self.prefix_idx_s3.get(key4, set())
        candidates |= self.prefix_idx_s3.get(key3, set())

        # 2. ZIP/PIN blocking
        zipcode = extract_zipcode(addr)
        if zipcode:
            candidates |= self.zip_idx_s2.get(zipcode, set())
            candidates |= self.zip_idx_s3.get(zipcode, set())

        # 3. Soundex blocking
        fw = get_first_token(name)
        if fw:
            sx = soundex(fw)
            candidates |= self.soundex_idx_s2.get(f"{cc}_{sx}", set())
            candidates |= self.soundex_idx_s3.get(f"{cc}_{sx}", set())
            # First-word exact
            candidates |= self.firstword_idx_s2.get(f"{cc}_{fw}", set())
            candidates |= self.firstword_idx_s3.get(f"{cc}_{fw}", set())

        # 4. TF-IDF retrieval
        if self.tfidf_s2:
            tfidf_s2_cands = self.tfidf_s2.query(name, top_k=self.top_k_tfidf)
            candidates |= set(tfidf_s2_cands)
        if self.tfidf_s3:
            tfidf_s3_cands = self.tfidf_s3.query(name, top_k=self.top_k_tfidf)
            candidates |= set(tfidf_s3_cands)

        return candidates

    def generate_candidates(self, s1: pd.DataFrame) -> dict:
        """Generate candidate dict {s1_entity_id -> set of candidate IDs}."""
        print(f"\n[Blocker] Generating candidates for {len(s1):,} S1 entities...")
        results = {}
        for _, row in tqdm(s1.iterrows(), total=len(s1), desc="  Blocking", ncols=80):
            eid = row['entity_id']
            cands = self.get_candidates(row)
            results[eid] = cands
        
        total_cands = sum(len(v) for v in results.values())
        print(f"  Total candidates generated: {total_cands:,}")
        print(f"  Avg candidates per S1 entity: {total_cands/len(s1):.1f}")
        return results


if __name__ == '__main__':
    # Quick test
    DTYPES = {'entity_id': str, 'business_name': str, 'business_address': str, 'country': str}
    print("Loading small samples for blocking test...")
    s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=DTYPES, nrows=10000)
    s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=DTYPES, nrows=10000)
    s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=DTYPES, nrows=100)
    
    blocker = EntityBlocker(top_k_tfidf=10)
    blocker.fit(s2, s3, use_tfidf=True)
    
    candidates = blocker.generate_candidates(s1)
    print(f"\nSample candidates for first S1 entity:")
    eid = list(candidates.keys())[0]
    print(f"  {eid}: {len(candidates[eid])} candidates")
    print(f"  Sample: {list(candidates[eid])[:5]}")
