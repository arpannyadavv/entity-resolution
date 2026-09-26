"""
Step 3: Feature Engineering (v2 - Enhanced)
=============================================
Computes similarity features between a pair (S1_entity, candidate_entity).

v2 additions:
  - Char bigram / trigram / 4-gram Jaccard   ← handles abbreviations
  - Token-subset score                        ← one name subset of other
  - Acronym similarity                        ← HDFC vs Housing Dev Fin Corp
  - Longest Common Substring ratio            ← LCS
  - Address number overlap                    ← street-number matching
  - Sorted-token similarity
  - 5 new meta / derived features

Total: ~40 features (up from 25)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import jellyfish
import numpy as np
from rapidfuzz import fuzz as rfuzz

from preprocess import (
    clean_name, clean_address, extract_zipcode,
    extract_tokens, get_first_token,
    get_char_ngrams, get_name_chars, extract_numbers
)


# ---------------------------------------------------------------------------
# Soundex (local copy to avoid circular import)
# ---------------------------------------------------------------------------
def _soundex(name: str) -> str:
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
# Basic helpers
# ---------------------------------------------------------------------------

def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_ngram_jaccard(a: str, b: str, n: int) -> float:
    """Jaccard similarity between char n-gram sets (spaces removed)."""
    sa = set(get_char_ngrams(get_name_chars(a), n))
    sb = set(get_char_ngrams(get_name_chars(b), n))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def shared_token_ratio(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    shared = len(ta & tb)
    mx = max(len(ta), len(tb))
    return shared / mx if mx > 0 else 0.0


def length_diff_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la + lb == 0:
        return 0.0
    return abs(la - lb) / max(la, lb)


def soundex_match(a: str, b: str) -> float:
    fa, fb = get_first_token(a), get_first_token(b)
    if not fa or not fb:
        return 0.0
    return 1.0 if _soundex(fa) == _soundex(fb) else 0.0


def token_subset_scores(a: str, b: str) -> tuple:
    """
    Returns (frac of a's tokens in b, frac of b's tokens in a, max of both).
    Captures cases where one name is an abbreviation / subset of the other.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0, 0.0, 0.0
    a_in_b = len(ta & tb) / len(ta)
    b_in_a = len(ta & tb) / len(tb)
    return a_in_b, b_in_a, max(a_in_b, b_in_a)


def acronym_similarity(a: str, b: str) -> float:
    """
    Compare the acronym formed from first letters of each token.
    E.g. 'Housing Development Finance Corp' → 'hdfc'
    """
    acro_a = ''.join(t[0] for t in a.split() if t)
    acro_b = ''.join(t[0] for t in b.split() if t)
    if not acro_a or not acro_b:
        return 0.0
    # Exact acronym match
    if acro_a == acro_b:
        return 1.0
    # One acronym is prefix of the other
    mn = min(len(acro_a), len(acro_b))
    if mn >= 2 and (acro_a.startswith(acro_b[:mn]) or acro_b.startswith(acro_a[:mn])):
        return mn / max(len(acro_a), len(acro_b))
    return 0.0


def lcs_ratio(a: str, b: str) -> float:
    """
    Longest Common Substring ratio relative to min length.
    Computed on space-stripped strings.
    """
    a = re.sub(r'\s+', '', a)
    b = re.sub(r'\s+', '', b)
    if not a or not b:
        return 0.0
    mn_len = min(len(a), len(b))
    mx_len = max(len(a), len(b))
    # DP
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best / mx_len if mx_len > 0 else 0.0


def number_overlap(addr1: str, addr2: str) -> float:
    """Fraction of numbers shared between two addresses."""
    n1 = set(extract_numbers(addr1))
    n2 = set(extract_numbers(addr2))
    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0
    return len(n1 & n2) / len(n1 | n2)


def sorted_token_similarity(a: str, b: str) -> float:
    """Jaccard on sorted-token representation of the name."""
    ta = sorted(a.split())
    tb = sorted(b.split())
    joined_a = ' '.join(ta)
    joined_b = ' '.join(tb)
    return rfuzz.ratio(joined_a, joined_b) / 100.0


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def compute_name_features(name1: str, name2: str) -> dict:
    """All name similarity features between two business names."""
    n1 = clean_name(name1) if isinstance(name1, str) else ''
    n2 = clean_name(name2) if isinstance(name2, str) else ''

    features = {}

    # ---- Rapidfuzz features ----
    features['name_levenshtein']       = rfuzz.ratio(n1, n2) / 100.0
    features['name_token_sort_ratio']  = rfuzz.token_sort_ratio(n1, n2) / 100.0
    features['name_token_set_ratio']   = rfuzz.token_set_ratio(n1, n2) / 100.0
    features['name_partial_ratio']     = rfuzz.partial_ratio(n1, n2) / 100.0
    features['name_wRatio']            = rfuzz.WRatio(n1, n2) / 100.0

    # ---- Jellyfish ----
    try:
        features['name_jaro_winkler'] = jellyfish.jaro_winkler_similarity(n1, n2)
    except Exception:
        features['name_jaro_winkler'] = 0.0

    # ---- Token-based ----
    features['name_jaccard']            = token_jaccard(n1, n2)
    features['name_shared_token_ratio'] = shared_token_ratio(n1, n2)
    features['name_len_diff']           = length_diff_ratio(n1, n2)
    features['name_soundex_match']      = soundex_match(n1, n2)

    # ---- Token counts ----
    t1, t2 = set(n1.split()), set(n2.split())
    features['name_shared_tokens'] = len(t1 & t2)
    features['name_max_tokens']    = max(len(t1), len(t2))
    features['name_min_tokens']    = min(len(t1), len(t2))

    # ---- Char n-gram Jaccard (NEW) ----
    features['name_char2_jaccard'] = char_ngram_jaccard(n1, n2, 2)
    features['name_char3_jaccard'] = char_ngram_jaccard(n1, n2, 3)
    features['name_char4_jaccard'] = char_ngram_jaccard(n1, n2, 4)

    # ---- Token subset (NEW) ----
    a_in_b, b_in_a, max_sub = token_subset_scores(n1, n2)
    features['name_subset_a_in_b'] = a_in_b
    features['name_subset_b_in_a'] = b_in_a
    features['name_subset_max']    = max_sub

    # ---- Acronym similarity (NEW) ----
    features['name_acronym_sim'] = acronym_similarity(n1, n2)

    # ---- LCS ratio (NEW) ----
    features['name_lcs_ratio'] = lcs_ratio(n1, n2)

    # ---- Sorted token similarity (NEW) ----
    features['name_sorted_token_sim'] = sorted_token_similarity(n1, n2)

    # ---- First token match (NEW) ----
    fw1 = get_first_token(name1) if isinstance(name1, str) else ''
    fw2 = get_first_token(name2) if isinstance(name2, str) else ''
    features['name_first_token_exact'] = 1.0 if (fw1 and fw2 and fw1 == fw2) else 0.0
    features['name_first_token_sim']   = rfuzz.ratio(fw1, fw2) / 100.0 if (fw1 and fw2) else 0.0

    return features


def compute_address_features(addr1: str, addr2: str) -> dict:
    """All address similarity features."""
    a1 = clean_address(addr1) if isinstance(addr1, str) else ''
    a2 = clean_address(addr2) if isinstance(addr2, str) else ''

    features = {}

    # Missing flags
    a1_miss = 1 if not a1 else 0
    a2_miss = 1 if not a2 else 0
    features['addr_a1_missing']   = a1_miss
    features['addr_a2_missing']   = a2_miss
    features['addr_both_missing'] = 1 if (a1_miss and a2_miss) else 0

    if not a1 or not a2:
        features['addr_levenshtein']       = 0.0
        features['addr_token_sort_ratio']  = 0.0
        features['addr_token_set_ratio']   = 0.0
        features['addr_partial_ratio']     = 0.0
        features['addr_jaccard']           = 0.0
        features['addr_shared_token_ratio']= 0.0
        features['addr_len_diff']          = 0.0
        features['addr_zip_match']         = 0
        features['addr_zip_both_missing']  = 1
        features['addr_shared_tokens']     = 0
        features['addr_number_overlap']    = 0.0
        features['addr_char3_jaccard']     = 0.0
        return features

    features['addr_levenshtein']       = rfuzz.ratio(a1, a2) / 100.0
    features['addr_token_sort_ratio']  = rfuzz.token_sort_ratio(a1, a2) / 100.0
    features['addr_token_set_ratio']   = rfuzz.token_set_ratio(a1, a2) / 100.0
    features['addr_partial_ratio']     = rfuzz.partial_ratio(a1, a2) / 100.0
    features['addr_jaccard']           = token_jaccard(a1, a2)
    features['addr_shared_token_ratio']= shared_token_ratio(a1, a2)
    features['addr_len_diff']          = length_diff_ratio(a1, a2)

    # ZIP
    z1 = extract_zipcode(addr1)
    z2 = extract_zipcode(addr2)
    features['addr_zip_match']        = 1 if (z1 and z2 and z1 == z2) else 0
    features['addr_zip_both_missing'] = 1 if (not z1 and not z2) else 0

    # Shared address tokens
    ta, tb = set(a1.split()), set(a2.split())
    features['addr_shared_tokens'] = len(ta & tb)

    # Address number overlap (NEW)
    features['addr_number_overlap'] = number_overlap(addr1, addr2)

    # Char 3-gram jaccard on address (NEW)
    features['addr_char3_jaccard'] = char_ngram_jaccard(a1, a2, 3)

    return features


def compute_meta_features(country1: str, country2: str) -> dict:
    """Country-level meta features."""
    c1 = str(country1).lower().strip() if isinstance(country1, str) else ''
    c2 = str(country2).lower().strip() if isinstance(country2, str) else ''
    return {
        'meta_country_match':    1 if c1 == c2 else 0,
        'meta_country_mismatch': 0 if c1 == c2 else 1,
    }


def compute_pair_features(s1_row: dict, cand_row: dict) -> dict:
    """Compute ALL features for a (S1, candidate) pair."""
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
        {'business_name': 'Test Corp', 'business_address': '123 Main St', 'country': 'US'},
        {'business_name': 'Test Corp', 'business_address': '123 Main St', 'country': 'US'}
    )
    return list(dummy.keys())


if __name__ == '__main__':
    pair = {
        's1': {'business_name': 'Prabhav Business Center Pvt. Ltd.',
               'business_address': '797 Lake Town Block A, Kolkata', 'country': 'India'},
        'cand': {'business_name': 'PRABHAV BUS CTR PRIVATE LTD',
                 'business_address': 'Lake Town, Howrah, West Bengal', 'country': 'India'},
    }
    feats = compute_pair_features(pair['s1'], pair['cand'])
    print(f"Total features: {len(feats)}")
    print("\nValues:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
