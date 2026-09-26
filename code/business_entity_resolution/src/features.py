"""
Step 3: Feature Engineering
Computes similarity features between a pair (S1_entity, candidate_entity).

Features:
  Name features:
    - Token Jaccard similarity
    - Levenshtein ratio (rapidfuzz)
    - Token sort ratio (rapidfuzz)
    - Partial ratio (rapidfuzz)
    - Jaro-Winkler (jellyfish)
    - Soundex match
    - Shared token count / max token count
    - Length difference ratio

  Address features:
    - Token Jaccard similarity
    - Levenshtein ratio on address
    - ZIP/PIN match (exact)
    - City/state token overlap
    - Address length difference

  Meta features:
    - Country exact match
    - Country mismatch flag
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import math
import jellyfish
import pandas as pd
import numpy as np
from rapidfuzz import fuzz as rfuzz
import re as _re

def soundex(name: str) -> str:
    """Simple Soundex encoding for phonetic blocking."""
    name = _re.sub(r'[^a-z]', '', name.lower())
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

from preprocess import (
    clean_name, clean_address, extract_zipcode,
    extract_tokens, get_first_token
)


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def shared_token_ratio(a: str, b: str) -> float:
    """Ratio of shared tokens to max token count."""
    ta, tb = set(a.split()), set(b.split())
    shared = len(ta & tb)
    max_len = max(len(ta), len(tb))
    return shared / max_len if max_len > 0 else 0.0


def length_diff_ratio(a: str, b: str) -> float:
    """Normalized length difference (0=same length, 1=max diff)."""
    la, lb = len(a), len(b)
    if la + lb == 0:
        return 0.0
    return abs(la - lb) / max(la, lb)


def soundex_match(a: str, b: str) -> float:
    """1.0 if soundex of first tokens match, 0.0 otherwise."""
    fa = get_first_token(a)
    fb = get_first_token(b)
    if not fa or not fb:
        return 0.0
    return 1.0 if soundex(fa) == soundex(fb) else 0.0



def compute_name_features(name1: str, name2: str) -> dict:
    """All name similarity features between two business names."""
    n1 = clean_name(name1) if isinstance(name1, str) else ''
    n2 = clean_name(name2) if isinstance(name2, str) else ''

    features = {}

    # Rapidfuzz features (0-100 scaled to 0-1)
    features['name_levenshtein'] = rfuzz.ratio(n1, n2) / 100.0
    features['name_token_sort_ratio'] = rfuzz.token_sort_ratio(n1, n2) / 100.0
    features['name_token_set_ratio'] = rfuzz.token_set_ratio(n1, n2) / 100.0
    features['name_partial_ratio'] = rfuzz.partial_ratio(n1, n2) / 100.0
    features['name_wRatio'] = rfuzz.WRatio(n1, n2) / 100.0

    # Jellyfish features
    try:
        features['name_jaro_winkler'] = jellyfish.jaro_winkler_similarity(n1, n2)
    except Exception:
        features['name_jaro_winkler'] = 0.0

    # Token-based features
    features['name_jaccard'] = token_jaccard(n1, n2)
    features['name_shared_token_ratio'] = shared_token_ratio(n1, n2)
    features['name_len_diff'] = length_diff_ratio(n1, n2)
    features['name_soundex_match'] = soundex_match(n1, n2)

    # Token count features
    t1 = set(n1.split())
    t2 = set(n2.split())
    features['name_shared_tokens'] = len(t1 & t2)
    features['name_max_tokens'] = max(len(t1), len(t2))
    features['name_min_tokens'] = min(len(t1), len(t2))

    return features


def compute_address_features(addr1: str, addr2: str) -> dict:
    """All address similarity features."""
    a1 = clean_address(addr1) if isinstance(addr1, str) else ''
    a2 = clean_address(addr2) if isinstance(addr2, str) else ''

    features = {}

    # Check for missing addresses
    a1_missing = 1 if not a1 else 0
    a2_missing = 1 if not a2 else 0
    features['addr_a1_missing'] = a1_missing
    features['addr_a2_missing'] = a2_missing
    features['addr_both_missing'] = 1 if (a1_missing and a2_missing) else 0

    if not a1 or not a2:
        # Can't compute similarity
        features['addr_levenshtein'] = 0.0
        features['addr_token_sort_ratio'] = 0.0
        features['addr_token_set_ratio'] = 0.0
        features['addr_partial_ratio'] = 0.0
        features['addr_jaccard'] = 0.0
        features['addr_shared_token_ratio'] = 0.0
        features['addr_len_diff'] = 0.0
        features['addr_zip_match'] = 0
        features['addr_zip_both_missing'] = 1
        features['addr_shared_tokens'] = 0
        return features

    features['addr_levenshtein'] = rfuzz.ratio(a1, a2) / 100.0
    features['addr_token_sort_ratio'] = rfuzz.token_sort_ratio(a1, a2) / 100.0
    features['addr_token_set_ratio'] = rfuzz.token_set_ratio(a1, a2) / 100.0
    features['addr_partial_ratio'] = rfuzz.partial_ratio(a1, a2) / 100.0
    features['addr_jaccard'] = token_jaccard(a1, a2)
    features['addr_shared_token_ratio'] = shared_token_ratio(a1, a2)
    features['addr_len_diff'] = length_diff_ratio(a1, a2)

    # ZIP matching
    z1 = extract_zipcode(addr1)
    z2 = extract_zipcode(addr2)
    zip_both_missing = 1 if (not z1 and not z2) else 0
    zip_match = 1 if (z1 and z2 and z1 == z2) else 0
    features['addr_zip_match'] = zip_match
    features['addr_zip_both_missing'] = zip_both_missing

    # Shared address tokens
    ta = set(a1.split())
    tb = set(a2.split())
    features['addr_shared_tokens'] = len(ta & tb)

    return features


def compute_meta_features(country1: str, country2: str) -> dict:
    """Country-level meta features."""
    c1 = str(country1).lower().strip() if isinstance(country1, str) else ''
    c2 = str(country2).lower().strip() if isinstance(country2, str) else ''
    return {
        'meta_country_match': 1 if c1 == c2 else 0,
        'meta_country_mismatch': 0 if c1 == c2 else 1,
    }


def compute_pair_features(s1_row: dict, cand_row: dict) -> dict:
    """Compute all features for a (S1, candidate) pair."""
    features = {}
    features.update(compute_name_features(
        s1_row.get('business_name', ''),
        cand_row.get('business_name', '')
    ))
    features.update(compute_address_features(
        s1_row.get('business_address', ''),
        cand_row.get('business_address', '')
    ))
    features.update(compute_meta_features(
        s1_row.get('country', ''),
        cand_row.get('country', '')
    ))
    return features


def get_feature_names() -> list:
    """Return ordered list of all feature names."""
    dummy = compute_pair_features(
        {'business_name': 'test', 'business_address': '123 main st', 'country': 'US'},
        {'business_name': 'test', 'business_address': '123 main st', 'country': 'US'}
    )
    return list(dummy.keys())


if __name__ == '__main__':
    # Test
    pair = {
        's1': {'business_name': 'Prabhav Business Center Pvt. Ltd.',
               'business_address': '797 Lake Town Block A, Kolkata', 'country': 'India'},
        'cand': {'business_name': 'PRABHAV BUSINESS CTR PRIVATE LTD',
                 'business_address': 'Lake Town, Howrah, West Bengal', 'country': 'India'},
    }
    feats = compute_pair_features(pair['s1'], pair['cand'])
    print("Feature names:", list(feats.keys()))
    print("\nValues:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
