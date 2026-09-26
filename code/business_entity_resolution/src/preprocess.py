"""
Step 1: Preprocessing & Normalization
Cleans and normalizes business names and addresses for Entity Resolution
"""

import re
import unicodedata

# ---- Legal suffix normalization ----
LEGAL_SUFFIXES = {
    r'\bincorporated\b': 'inc',
    r'\bcorporation\b': 'corp',
    r'\blimited\b': 'ltd',
    r'\bprivate\b': 'pvt',
    r'\bpvt\.?\b': 'pvt',
    r'\bltd\.?\b': 'ltd',
    r'\binc\.?\b': 'inc',
    r'\bcorp\.?\b': 'corp',
    r'\bllc\.?\b': 'llc',
    r'\bllp\.?\b': 'llp',
    r'\bco\.?\b': 'co',
    r'\bplc\.?\b': 'plc',
    r'\bgmbh\.?\b': 'gmbh',
    r'\bsarl\.?\b': 'sarl',
    r'\bsas\.?\b': 'sas',
    r'\bsa\.?\b': 'sa',
    r'\bsrl\.?\b': 'srl',
    r'\bspa\.?\b': 'spa',
    r'\band\b': '&',
    r'\bthe\b': '',
}

# Address abbreviations
ADDRESS_ABBR = {
    r'\bstreet\b': 'st',
    r'\broad\b': 'rd',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd',
    r'\bdrive\b': 'dr',
    r'\blane\b': 'ln',
    r'\bcourt\b': 'ct',
    r'\bplace\b': 'pl',
    r'\bsuite\b': 'ste',
    r'\bapartment\b': 'apt',
    r'\bfloor\b': 'fl',
    r'\bnorth\b': 'n',
    r'\bsouth\b': 's',
    r'\beast\b': 'e',
    r'\bwest\b': 'w',
    r'\bnortheast\b': 'ne',
    r'\bnorthwest\b': 'nw',
    r'\bsoutheast\b': 'se',
    r'\bsouthwest\b': 'sw',
}


def normalize_unicode(text: str) -> str:
    """Convert unicode to closest ASCII representation."""
    if not isinstance(text, str):
        return ''
    try:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', errors='ignore').decode('ascii')
    except Exception:
        text = str(text)
    return text


def clean_name(name: str) -> str:
    """
    Normalize business name:
    - Lowercase
    - Normalize unicode
    - Expand/collapse legal suffixes
    - Remove punctuation
    - Collapse whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    
    text = normalize_unicode(name)
    text = text.lower().strip()
    
    # Remove punctuation (keep & for now)
    text = re.sub(r'[^\w\s&]', ' ', text)
    
    # Normalize legal suffixes
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_address(address: str) -> str:
    """
    Normalize business address:
    - Lowercase
    - Normalize unicode
    - Abbreviate common terms
    - Remove punctuation
    - Collapse whitespace
    """
    if not isinstance(address, str) or not address.strip():
        return ''
    
    text = normalize_unicode(address)
    text = text.lower().strip()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize address abbreviations
    for pattern, replacement in ADDRESS_ABBR.items():
        text = re.sub(pattern, replacement, text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_tokens(text: str) -> set:
    """Split cleaned text into tokens (set)."""
    if not text:
        return set()
    return set(text.split())


def extract_name_tokens(name: str) -> set:
    """Extract meaningful tokens from business name (remove stopwords)."""
    STOP = {'the', 'a', 'an', 'of', 'in', 'at', 'for', 'and', '&', 'co', 'inc', 'ltd', 'pvt', 'llc', 'corp'}
    tokens = extract_tokens(clean_name(name))
    # Keep suffix-filtered tokens too for matching, only remove pure stopwords
    return tokens - STOP if len(tokens - STOP) > 0 else tokens


def get_first_token(name: str) -> str:
    """Get first meaningful token of business name for blocking."""
    tokens = clean_name(name).split()
    stopwords = {'the', 'a', 'an'}
    for t in tokens:
        if t not in stopwords and len(t) > 1:
            return t
    return tokens[0] if tokens else ''


def get_name_prefix(name: str, n: int = 3) -> str:
    """Get first n chars of cleaned name for prefix blocking."""
    cleaned = clean_name(name)
    return cleaned[:n] if len(cleaned) >= n else cleaned


def extract_zipcode(address: str) -> str:
    """Extract ZIP/PIN code from address."""
    if not isinstance(address, str):
        return ''
    # US ZIP: 5 digits optionally followed by -4
    us_zip = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    if us_zip:
        return us_zip.group(1)
    # India PIN: 6 digits
    india_pin = re.search(r'\b(\d{6})\b', address)
    if india_pin:
        return india_pin.group(1)
    # France postal: 5 digits
    fr_postal = re.search(r'\b(\d{5})\b', address)
    if fr_postal:
        return fr_postal.group(1)
    return ''


def extract_city_state(address: str) -> str:
    """Rough extraction of city/state from address (last 2 comma-parts)."""
    if not isinstance(address, str):
        return ''
    parts = [p.strip() for p in address.split(',')]
    if len(parts) >= 2:
        return clean_address(','.join(parts[-2:]))
    return clean_address(address)


def get_blocking_key(name: str, country: str, prefix_len: int = 4) -> str:
    """
    Generate a blocking key combining country + name prefix.
    This is the primary blocking key.
    """
    country_key = str(country).lower().strip()[:2] if isinstance(country, str) else 'xx'
    name_prefix = get_name_prefix(name, prefix_len)
    return f"{country_key}_{name_prefix}"


if __name__ == '__main__':
    # Quick test
    tests = [
        ("Prabhav Business Center Pvt. Ltd.", "B-47 Lake Town, Kolkata"),
        ("PRABHAV BUSINESS CTR PRIVATE LIMITED", "Lake Town Block A, Howrah"),
        ("McDonald's Corporation", "1 McDonald's Plaza, Oak Brook, IL 60523"),
        ("McDonalds Corp", "Oak Brook IL"),
    ]
    for name, addr in tests:
        print(f"Raw:     {name} | {addr}")
        print(f"Cleaned: {clean_name(name)} | {clean_address(addr)}")
        print(f"Key:     {get_blocking_key(name, 'US')}")
        print()
