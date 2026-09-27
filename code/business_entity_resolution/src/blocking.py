"""
Step 2: Blocking / Candidate Generation (v2.1 - Ultra-Fast Multi-Strategy)
========================================================================
High-recall, memory-efficient inverted index blocking:
  1. Country-prefixed Name Prefixes (3, 4, 5 chars) with variant normalization
  2. Distinctive Name Tokens (inverted index over informative terms)
  3. First-Word Soundex (phonetic matching for misspellings)
  4. First-Word Exact Match
  5. Sorted-Token Key (word-order invariant matching)
  6. Address Physical Blocking (ZIP/PIN code + Street Number & Street Name)
  7. Multi-key Hit Ranking: Candidates matching multiple signals ranked higher
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import math
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from preprocess import (
    clean_name, clean_address, get_blocking_key,
    get_name_prefix, extract_zipcode, get_first_token,
    extract_tokens, get_char_ngrams, get_name_chars,
    get_sorted_token_key, normalize_name_variants,
    get_addr_keys
)

COMMON_NAME_TOKENS = {
    'corp', 'limited', 'services', 'technologies', 'solutions',
    'enterprises', 'enterprise', 'industries', 'group', 'international',
    'national', 'products', 'management', 'consulting', 'holdings',
    'company', 'trading', 'general', 'global', 'systems'
}


def soundex(name: str) -> str:
    """Standard Soundex encoding."""
    name = re.sub(r'[^a-z]', '', str(name).lower())
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


class EntityBlocker:
    """
    High-performance multi-strategy blocker with inverted index lookup
    and multi-signal candidate ranking.
    """

    def __init__(self, max_candidates: int = 25, max_block_size: int = 300,
                 top_k_tfidf: int = 25, top_k_char: int = 20):
        self.max_candidates = max_candidates
        self.max_block_size = max_block_size
        self.top_k_tfidf = top_k_tfidf
        self.top_k_char = top_k_char
        self.index = defaultdict(list)

    def _extract_keys(self, name: str, addr: str, country: str) -> list:
        cc = str(country).lower().strip()[:2] if country else 'xx'
        keys = []

        # 1. Name variants prefixes & tokens
        variants = normalize_name_variants(name)
        for nvar in variants:
            # Prefixes
            if len(nvar) >= 3:
                keys.append(('pfx3', f"p3_{cc}_{nvar[:3]}", 1))
            if len(nvar) >= 4:
                keys.append(('pfx4', f"p4_{cc}_{nvar[:4]}", 2))
            if len(nvar) >= 5:
                keys.append(('pfx5', f"p5_{cc}_{nvar[:5]}", 3))

            # Phonetic soundex of first word
            fw = get_first_token(nvar)
            if fw:
                keys.append(('fw', f"fw_{cc}_{fw}", 3))
                keys.append(('sx', f"sx_{cc}_{soundex(fw)}", 1))

            # Distinctive tokens
            for w in nvar.split():
                if len(w) >= 4 and w not in COMMON_NAME_TOKENS:
                    keys.append(('tok', f"tok_{cc}_{w}", 3))

        # 2. Sorted-token key (word-order invariant)
        stk = get_sorted_token_key(name, country)
        if stk:
            keys.append(('stk', f"stk_{stk}", 3))

        # 3. Address keys (ZIP / street number + street token)
        for ak in get_addr_keys(addr, country):
            keys.append(('addr', ak, 3))

        return keys

    def fit(self, s2: pd.DataFrame, s3: pd.DataFrame, use_tfidf: bool = False):
        """Build all inverted indexes from S2 and S3."""
        print(f"\n[Blocker] Indexing S2 ({len(s2):,} records) and S3 ({len(s3):,} records)...")
        self.index.clear()

        for source_name, dataset in [('S2', s2), ('S3', s3)]:
            if hasattr(dataset, 'itertuples'):
                iterable = dataset.itertuples(index=False)
                total = len(dataset)
            else:
                iterable = dataset
                total = len(dataset)

            for row in tqdm(iterable, total=total, desc=f"  Indexing {source_name}", ncols=80):
                eid = row[0]
                name = str(row[1]) if (row[1] is not None and pd.notna(row[1])) else ''
                addr = str(row[2]) if (row[2] is not None and pd.notna(row[2])) else ''
                country = str(row[3]) if (len(row) > 3 and row[3] is not None and pd.notna(row[3])) else ''

                extracted = self._extract_keys(name, addr, country)
                seen_keys = set()
                for _, key_str, _ in extracted:
                    if key_str not in seen_keys:
                        seen_keys.add(key_str)
                        lst = self.index[key_str]
                        if len(lst) <= self.max_block_size:
                            lst.append(eid)

        # Prune excessively large blocks (stop words / noisy prefixes)
        pruned = 0
        total_keys = len(self.index)
        for k in list(self.index.keys()):
            if len(self.index[k]) > self.max_block_size:
                del self.index[k]
                pruned += 1
        print(f"  Total keys: {total_keys:,} | Pruned {pruned:,} large blocks (> {self.max_block_size})")

    def get_candidates(self, s1_row) -> set:
        """Get top candidates for a single S1 entity, ranked by multi-signal match count."""
        if isinstance(s1_row, (list, tuple)):
            name = str(s1_row[1]) if len(s1_row) > 1 and s1_row[1] is not None and pd.notna(s1_row[1]) else ''
            addr = str(s1_row[2]) if len(s1_row) > 2 and s1_row[2] is not None and pd.notna(s1_row[2]) else ''
            country = str(s1_row[3]) if len(s1_row) > 3 and s1_row[3] is not None and pd.notna(s1_row[3]) else ''
        else:
            name = str(s1_row['business_name']) if pd.notna(s1_row['business_name']) else ''
            addr = str(s1_row['business_address']) if pd.notna(s1_row['business_address']) else ''
            country = str(s1_row['country']) if pd.notna(s1_row['country']) else ''

        extracted = self._extract_keys(name, addr, country)
        hits = defaultdict(int)
        seen_keys = set()

        for _, key_str, weight in extracted:
            if key_str in seen_keys or key_str not in self.index:
                continue
            seen_keys.add(key_str)
            for cid in self.index[key_str]:
                hits[cid] += weight

        if not hits:
            return set()

        # Sort by match weight descending
        sorted_cands = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
        return {c[0] for c in sorted_cands}

    def generate_candidates(self, s1: pd.DataFrame) -> dict:
        """Generate candidate dictionary {s1_entity_id -> set of candidate IDs}."""
        print(f"\n[Blocker] Generating candidates for {len(s1):,} S1 entities...")
        results = {}
        for _, row in tqdm(s1.iterrows(), total=len(s1), desc="  Blocking", ncols=80):
            eid = row['entity_id']
            results[eid] = self.get_candidates(row)

        total_cands = sum(len(v) for v in results.values())
        print(f"  Total candidates generated: {total_cands:,}")
        print(f"  Avg per S1 entity: {total_cands / len(s1):.1f}")
        return results
