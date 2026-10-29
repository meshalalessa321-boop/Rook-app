#!/usr/bin/env python3
"""Rook byte-level BPE tokenizer: dependency-free, learned only from Rook's own corpus.

Why this exists: a byte-level model spends most of its small capacity learning to
spell. Merging frequent byte sequences into tokens lets the same 256-token context
cover about four times as much text.

Rules this file follows:
  * The merges are learned from the project's own documented training text. No
    pretrained tokenizer or vocabulary is downloaded or copied.
  * Nothing here adds an identity, role, instruction or canned text.
  * Every byte sequence round-trips exactly (encode then decode returns the input).

Token ids: 0..255 are raw bytes, 256..255+n are learned merges (in learning order),
then three special ids: PAD, BOS, EOS. With n = 0 this is exactly the old byte model
(PAD=256, BOS=257, EOS=258, vocab=259), so older checkpoints stay compatible.

rook-tokenizer.js implements the same split and merge rules for the browser. The two
are checked against each other by test_tokenizer_js.py; change both together.
"""
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict

FORMAT = "rook-bytebpe-v1"

# Pre-tokenization (on raw bytes). Same shape as the GPT-2 split, without Unicode
# classes so Python and JavaScript agree exactly:
#   " ?" + run of letters/digits/high bytes     (high bytes = UTF-8 multibyte text)
#   " ?" + run of anything else that is not whitespace
#   whitespace runs, leaving the last space free to attach to the next word
PRETOKEN = re.compile(
    rb" ?[0-9A-Za-z\x80-\xff]+| ?[^\s0-9A-Za-z\x80-\xff]+|\s+(?!\S)|\s+"
)


def pretokens(data):
    """Split bytes into pre-tokens. They always concatenate back to `data`."""
    return PRETOKEN.findall(data)


class ByteBPE:
    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.n = len(self.merges)
        self.pad = 256 + self.n
        self.bos = 257 + self.n
        self.eos = 258 + self.n
        self.vocab_size = 259 + self.n
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.table = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.table.append(self.table[a] + self.table[b])
        self._cache = {}

    # ----- encoding / decoding -------------------------------------------------
    def _encode_piece(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece)
        while len(ids) > 1:
            best_rank, best_pair = None, None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, (ids[i], ids[i + 1])
            if best_pair is None:
                break
            new_id = 256 + best_rank
            merged, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if len(self._cache) < 500_000:
            self._cache[piece] = ids
        return ids

    def encode(self, data):
        """Encode bytes (or str) to token ids. Never emits special ids."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        out = []
        for piece in pretokens(data):
            out.extend(self._encode_piece(piece))
        return out

    def decode_bytes(self, ids):
        return b"".join(self.table[i] for i in ids if i < 256 + self.n)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ----- (de)serialization ---------------------------------------------------
    def digest(self):
        return hashlib.sha256(json.dumps(self.merges, separators=(",", ":")).encode("ascii")).hexdigest()

    def to_json(self):
        return {"format": FORMAT, "vocab_size": self.vocab_size,
                "pad": self.pad, "bos": self.bos, "eos": self.eos,
                "merges": [list(pair) for pair in self.merges], "sha256": self.digest()}

    @classmethod
    def from_json(cls, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        if obj.get("format") != FORMAT:
            raise ValueError(f"Unsupported tokenizer format: {obj.get('format')!r}")
        tok = cls(obj["merges"])
        if (obj.get("vocab_size"), obj.get("pad"), obj.get("bos"), obj.get("eos")) != \
                (tok.vocab_size, tok.pad, tok.bos, tok.eos):
            raise ValueError("Tokenizer special ids do not match its merge count.")
        if obj.get("sha256") and obj["sha256"] != tok.digest():
            raise ValueError("Tokenizer checksum mismatch.")
        return tok


def byte_tokenizer():
    """The original 259-token byte vocabulary (no merges)."""
    return ByteBPE([])


def train_bpe(texts, vocab_size=4096, max_bytes=30_000_000):
    """Learn merges from an iterable of str/bytes documents.

    Deterministic: the same texts in the same order always give the same merges.
    Reads at most `max_bytes` bytes (whole documents) to keep training fast.
    """
    n_merges = vocab_size - 259
    if n_merges < 0:
        raise ValueError("vocab_size must be at least 259")
    counts, used = Counter(), 0
    for text in texts:
        data = text.encode("utf-8") if isinstance(text, str) else text
        counts.update(pretokens(data))
        used += len(data)
        if used >= max_bytes:
            break
    # Words: pre-token -> (symbol list, frequency). Sorted for determinism.
    words = [[list(piece), freq] for piece, freq in sorted(counts.items())]
    pair_count = defaultdict(int)
    where = defaultdict(set)
    for wi, (symbols, freq) in enumerate(words):
        for pair in zip(symbols, symbols[1:]):
            pair_count[pair] += freq
            where[pair].add(wi)
    # Max-heap on (count, then smallest pair) with lazy deletion.
    heap = [(-c, p) for p, c in pair_count.items() if c > 0]
    heapq.heapify(heap)
    merges = []
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        if pair_count.get(pair, 0) != -neg or -neg <= 0:
            continue  # stale entry
        new_id = 256 + len(merges)
        merges.append(pair)
        touched = {}
        for wi in sorted(where[pair]):
            symbols, freq = words[wi]
            # Remove this word's old pair counts, merge, add the new ones.
            for p in zip(symbols, symbols[1:]):
                pair_count[p] -= freq
                touched[p] = True
            merged, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            words[wi][0] = merged
            for p in zip(merged, merged[1:]):
                pair_count[p] += freq
                where[p].add(wi)
                touched[p] = True
        for p in touched:
            if pair_count[p] > 0:
                heapq.heappush(heap, (-pair_count[p], p))
        pair_count.pop(pair, None)
        where.pop(pair, None)
    return ByteBPE(merges)


def train_bpe_reference(texts, vocab_size):
    """Slow, obviously-correct trainer used only by tests to check train_bpe."""
    n_merges = vocab_size - 259
    counts = Counter()
    for text in texts:
        data = text.encode("utf-8") if isinstance(text, str) else text
        counts.update(pretokens(data))
    words = [[list(piece), freq] for piece, freq in sorted(counts.items())]
    merges = []
    while len(merges) < n_merges:
        pairs = Counter()
        for symbols, freq in words:
            for p in zip(symbols, symbols[1:]):
                pairs[p] += freq
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        new_id = 256 + len(merges)
        merges.append(best)
        for word in words:
            symbols, merged, i = word[0], [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            word[0] = merged
    return ByteBPE(merges)
