"""
Step 1: Preprocessing & Normalization (v2 - Enhanced)
=======================================================
- Multi-language legal suffix normalization (EN / IN / FR / DE / IT / ES)
- Character n-gram helpers for blocking
- Better address abbreviation normalization
- Number extraction helpers
"""

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Legal suffix lookup (order matters: longer patterns first)
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = [
    # ---- Long-form patterns first (must come before short abbrevs) ----
    (r'\bprivate\s+limited\b',  'pvt ltd'),
    (r'\bpvt\s+ltd\.?\b',       'pvt ltd'),
    # ---- English ----
    (r'\bincorporated\b',  'inc'),
    (r'\bcorporation\b',   'corp'),
    (r'\blimited\b',       'ltd'),
    (r'\bprivate\b',       'pvt'),
    (r'\bpvt\.?\b',        'pvt'),
    (r'\bltd\.?\b',        'ltd'),
    (r'\binc\.?\b',        'inc'),
    (r'\bcorp\.?\b',       'corp'),
    (r'\bllc\.?\b',        'llc'),
    (r'\bllp\.?\b',        'llp'),
    (r'\bplc\.?\b',        'plc'),
    (r'\bco\.?\b',         'co'),
    (r'\bbros\.?\b',       'bros'),
    (r'\bbrothers\b',      'bros'),
    # ---- French ----
    (r'\bsociete\b',       'soc'),
    (r'\bsociété\b',       'soc'),
    (r'\bcompagnie\b',     'cie'),
    (r'\bsarl\.?\b',       'sarl'),
    (r'\bsas\.?\b',        'sas'),
    (r'\beurl\.?\b',       'eurl'),
    (r'\bsnc\.?\b',        'snc'),
    (r'\bsci\.?\b',        'sci'),
    (r'\bscp\.?\b',        'scp'),
    (r'\bsa\.?\b',         'sa'),
    # ---- German ----
    (r'\bgmbh\.?\b',       'gmbh'),
    (r'\bag\.?\b',         'ag'),
    (r'\bkg\.?\b',         'kg'),
    # ---- Italian / Spanish ----
    (r'\bsrl\.?\b',        'srl'),
    (r'\bspa\.?\b',        'spa'),
    (r'\bslu\.?\b',        'slu'),
    (r'\bsl\.?\b',         'sl'),
    # ---- Connectors ----
    (r'\band\b',           '&'),
    (r'\bthe\b',           ''),
    # ---- Common business words (normalise to short forms) ----
    (r'\benterprises\b',   'ent'),
    (r'\benterprise\b',    'ent'),
    (r'\btrading\b',       'trdg'),
    (r'\bindustries\b',    'ind'),
    (r'\bindustry\b',      'ind'),
    (r'\bservices\b',      'svc'),
    (r'\bservice\b',       'svc'),
    (r'\bgroup\b',         'grp'),
    (r'\binternational\b', 'intl'),
    (r'\bnational\b',      'natl'),
    (r'\bsolutions\b',     'sol'),
    (r'\btechnologies\b',  'tech'),
    (r'\btechnology\b',    'tech'),
    (r'\bmanufacturing\b', 'mfg'),
    (r'\bdistributors\b',  'dist'),
    (r'\bdistributor\b',   'dist'),
    (r'\bsuppliers\b',     'supp'),
    (r'\bsupplier\b',      'supp'),
    (r'\bassociates\b',    'assoc'),
    (r'\bassociate\b',     'assoc'),
    (r'\bconsultants\b',   'consult'),
    (r'\bconsultant\b',    'consult'),
    (r'\bventures\b',      'vent'),
    (r'\bventure\b',       'vent'),
    (r'\bholdings\b',      'hldg'),
    (r'\bholding\b',       'hldg'),
    (r'\bexports\b',       'exp'),
    (r'\bexport\b',        'exp'),
    (r'\bimports\b',       'imp'),
    (r'\bimport\b',        'imp'),
    (r'\bcentres\b',       'ctr'),
    (r'\bcentre\b',        'ctr'),
    (r'\bcenters\b',       'ctr'),
    (r'\bcenter\b',        'ctr'),
    (r'\bmedical\b',       'med'),
    (r'\bhospital\b',      'hosp'),
    (r'\bclinic\b',        'clin'),
    (r'\brestaurant\b',    'rest'),
    (r'\bhotels\b',        'htl'),
    (r'\bhotel\b',         'htl'),
    (r'\bshops\b',         'shp'),
    (r'\bshop\b',          'shp'),
    (r'\bstores\b',        'str'),
    (r'\bstore\b',         'str'),
    (r'\bmart\b',          'mrt'),
    (r'\bmarket\b',        'mkt'),
    (r'\bwholesale\b',     'whsl'),
    (r'\bretail\b',        'ret'),
    (r'\bagencies\b',      'agcy'),
    (r'\bagency\b',        'agcy'),
    (r'\bworks\b',         'wks'),
    (r'\binfrastructure\b', 'infra'),
    (r'\bfinance\b',       'fin'),
    (r'\bfinancial\b',     'fin'),
    (r'\bfoundation\b',    'fdn'),
    (r'\bconstruction\b',  'const'),
    (r'\brealty\b',        'rlty'),
    (r'\bproperties\b',    'prop'),
    (r'\bproperty\b',      'prop'),
    (r'\bdevelopers\b',    'dev'),
    (r'\bdeveloper\b',     'dev'),
    (r'\bdevelopment\b',   'dev'),
]

# Compile once at module load
_SUFFIX_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in LEGAL_SUFFIXES]

# ---------------------------------------------------------------------------
ADDRESS_ABBR = [
    (r'\bstreet\b',    'st'),
    (r'\broad\b',      'rd'),
    (r'\bavenue\b',    'ave'),
    (r'\bboulevard\b', 'blvd'),
    (r'\bdrive\b',     'dr'),
    (r'\blane\b',      'ln'),
    (r'\bcourt\b',     'ct'),
    (r'\bplace\b',     'pl'),
    (r'\bsuite\b',     'ste'),
    (r'\bapartment\b', 'apt'),
    (r'\bfloor\b',     'fl'),
    (r'\bnorth\b',     'n'),
    (r'\bsouth\b',     's'),
    (r'\beast\b',      'e'),
    (r'\bwest\b',      'w'),
    (r'\bnortheast\b', 'ne'),
    (r'\bnorthwest\b', 'nw'),
    (r'\bsoutheast\b', 'se'),
    (r'\bsouthwest\b', 'sw'),
    # French
    (r'\bimpasse\b',   'imp'),
    (r'\bchemin\b',    'ch'),
    (r'\bpassage\b',   'pass'),
    (r'\ballee\b',     'alle'),
    (r'\ballée\b',     'alle'),
    # Indian
    (r'\bnagar\b',     'ngr'),
    (r'\bcolony\b',    'col'),
    (r'\bsector\b',    'sec'),
    (r'\bphase\b',     'ph'),
    (r'\bblock\b',     'blk'),
    (r'\bbuilding\b',  'bldg'),
    (r'\bopposite\b',  'opp'),
    (r'\bnear\b',      ''),    # remove noise word
    (r'\bmain\b',      ''),    # remove noise word
]

_ADDR_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in ADDRESS_ABBR]


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=131072)
def normalize_unicode(text: str) -> str:
    """Convert unicode to closest ASCII."""
    if not isinstance(text, str):
        return ''
    try:
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', errors='ignore').decode('ascii')
    except Exception:
        text = str(text)
    return text


@lru_cache(maxsize=131072)
def clean_name(name: str) -> str:
    """
    Normalize business name:
      1. Unicode → ASCII
      2. Lowercase
      3. Remove punctuation (keep & and digits)
      4. Expand / collapse legal suffixes
      5. Collapse whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    text = normalize_unicode(name)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s&]', ' ', text)
    text = re.sub(r'\b(www|https?|http)\b', ' ', text)
    text = re.sub(r'\b(com|org|net|gov|edu|info)\b', ' ', text)
    for pat, repl in _SUFFIX_PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name_variants(name: str) -> list:
    """Generate canonical name variants:
    - Stripped of honorifics (M/s, Sri, Shri, Shree, Dr, etc.)
    - Split around DBA / AKA expressions (e.g. 'Korbrixx D.B.A. Obsidian LLC' -> 'obsidian llc')
    """
    c = clean_name(name)
    if not c:
        return []
    # Strip honorific prefixes
    c_stripped = re.sub(r'^(m\s*s\b|sri\b|shri\b|shree\b|dr\b|mr\b|prof\b)\s*', '', c)
    # Split around dba / aka
    parts = re.split(r'\b(dba|d\s*b\s*a|aka|t\s*a|trading\s+as)\b', c_stripped)
    variants = [c, c_stripped]
    if len(parts) > 1:
        variants.append(parts[-1].strip())
        variants.append(parts[0].strip())
    # Return unique non-empty variants
    seen = set()
    res = []
    for v in variants:
        v_clean = re.sub(r'\s+', ' ', v).strip()
        if v_clean and v_clean not in seen:
            seen.add(v_clean)
            res.append(v_clean)
    return res


def get_addr_keys(address: str, country: str) -> list:
    """Extract physical address blocking keys (ZIP code, street number + street token)."""
    if not isinstance(address, str) or not address.strip():
        return []
    cc = str(country).lower().strip()[:2] if isinstance(country, str) else 'xx'
    keys = []
    zc = extract_zipcode(address)
    if zc:
        keys.append(f"zip_{zc}")
    c_addr = clean_address(address)
    nums = extract_numbers(address)
    words = [w for w in c_addr.split() if len(w) >= 4 and not w.isdigit()]
    if nums and words:
        keys.append(f"addr_{cc}_{nums[0]}_{words[0]}")
        if len(words) > 1:
            keys.append(f"addr_{cc}_{nums[0]}_{words[1]}")
    return keys


@lru_cache(maxsize=131072)
def clean_address(address: str) -> str:
    """
    Normalize business address:
      1. Unicode → ASCII
      2. Lowercase
      3. Remove punctuation
      4. Abbreviate common terms
      5. Collapse whitespace
    """
    if not isinstance(address, str) or not address.strip():
        return ''
    text = normalize_unicode(address)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    for pat, repl in _ADDR_PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Character n-gram helpers (new in v2)
# ---------------------------------------------------------------------------

def get_name_chars(name: str) -> str:
    """Return cleaned name with all spaces removed (for char n-gram blocking)."""
    cleaned = clean_name(name)
    return re.sub(r'\s+', '', cleaned)


def get_char_ngrams(text: str, n: int) -> list:
    """Extract character n-grams from text."""
    text = re.sub(r'\s+', '', text)   # strip spaces
    if not text or len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def get_char_ngram_set(name: str, n: int) -> set:
    """Return set of char n-grams for a cleaned name (spaces removed)."""
    chars = get_name_chars(name)
    return set(get_char_ngrams(chars, n))


# ---------------------------------------------------------------------------
# Blocking key helpers
# ---------------------------------------------------------------------------

def extract_tokens(text: str) -> set:
    if not text:
        return set()
    return set(text.split())


def extract_name_tokens(name: str) -> set:
    STOP = {'the', 'a', 'an', 'of', 'in', 'at', 'for', 'and', '&',
            'co', 'inc', 'ltd', 'pvt', 'llc', 'corp'}
    tokens = extract_tokens(clean_name(name))
    return tokens - STOP if len(tokens - STOP) > 0 else tokens


def get_first_token(name: str) -> str:
    tokens = clean_name(name).split()
    stopwords = {'the', 'a', 'an'}
    for t in tokens:
        if t not in stopwords and len(t) > 1:
            return t
    return tokens[0] if tokens else ''


def get_name_prefix(name: str, n: int = 3) -> str:
    cleaned = clean_name(name)
    return cleaned[:n] if len(cleaned) >= n else cleaned


def extract_zipcode(address: str) -> str:
    if not isinstance(address, str):
        return ''
    # US ZIP: 5 digits optionally followed by -4
    m = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    if m:
        return m.group(1)
    # India PIN: 6 digits
    m = re.search(r'\b(\d{6})\b', address)
    if m:
        return m.group(1)
    # France postal / generic 5-digit
    m = re.search(r'\b(\d{5})\b', address)
    if m:
        return m.group(1)
    return ''


def extract_numbers(text: str) -> list:
    """Extract all numeric sequences (street numbers, etc.)."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\d+', text)


def extract_city_state(address: str) -> str:
    if not isinstance(address, str):
        return ''
    parts = [p.strip() for p in address.split(',')]
    if len(parts) >= 2:
        return clean_address(','.join(parts[-2:]))
    return clean_address(address)


def get_blocking_key(name: str, country: str, prefix_len: int = 4) -> str:
    country_key = str(country).lower().strip()[:2] if isinstance(country, str) else 'xx'
    name_prefix = get_name_prefix(name, prefix_len)
    return f"{country_key}_{name_prefix}"


def get_sorted_token_key(name: str, country: str) -> str:
    """Blocking key based on sorted tokens (handles word-order variations)."""
    cc = str(country).lower().strip()[:2] if isinstance(country, str) else 'xx'
    tokens = sorted(clean_name(name).split())
    key_str = '_'.join(tokens[:3])   # use first 3 sorted tokens
    return f"{cc}_{key_str}" if key_str else ''


if __name__ == '__main__':
    tests = [
        ("Prabhav Business Center Pvt. Ltd.", "B-47 Lake Town, Kolkata"),
        ("PRABHAV BUS CTR PRIVATE LIMITED", "Lake Town Block A, Howrah"),
        ("McDonald's Corporation", "1 McDonald's Plaza, Oak Brook, IL 60523"),
        ("SARL Boulangerie du Marché", "15, Rue de la Paix, 75001 Paris"),
    ]
    for name, addr in tests:
        print(f"Raw:     {name} | {addr}")
        print(f"Cleaned: {clean_name(name)} | {clean_address(addr)}")
        print(f"Chars:   {get_name_chars(name)}")
        print(f"3-grams: {get_char_ngram_set(name, 3)}")
        print()
