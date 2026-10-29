/* Rook byte-level BPE tokenizer for the browser (and Node, for tests).
 *
 * Must behave exactly like training/rook_tokenizer.py: same pre-token split, same
 * merge order, same ids. training/test_tokenizer_js.py checks the two against each
 * other. The merges come from the published tokenizer.json, learned only from
 * Rook's own training text. This file adds no identity, role, or scripted text.
 */
(function (root) {
  "use strict";

  const isWs = (b) => b === 32 || (b >= 9 && b <= 13);
  const isAlnum = (b) => (b >= 48 && b <= 57) || (b >= 65 && b <= 90) || (b >= 97 && b <= 122) || b >= 128;
  const isPunct = (b) => !isWs(b) && !isAlnum(b);

  // Split bytes into pre-tokens: same rules as the Python regex
  //   ` ?[0-9A-Za-z\x80-\xff]+| ?[^\s0-9A-Za-z\x80-\xff]+|\s+(?!\S)|\s+`
  function pretokenize(bytes) {
    const out = [];
    const n = bytes.length;
    let i = 0;
    while (i < n) {
      const c = bytes[i];
      let j;
      if (c === 32 && i + 1 < n && isAlnum(bytes[i + 1])) {
        j = i + 1;
        while (j < n && isAlnum(bytes[j])) j++;
      } else if (isAlnum(c)) {
        j = i;
        while (j < n && isAlnum(bytes[j])) j++;
      } else if (c === 32 && i + 1 < n && isPunct(bytes[i + 1])) {
        j = i + 1;
        while (j < n && isPunct(bytes[j])) j++;
      } else if (isPunct(c)) {
        j = i;
        while (j < n && isPunct(bytes[j])) j++;
      } else {
        // Whitespace run. At the end of input take all of it; otherwise leave the
        // last whitespace byte free (a space there attaches to the next word).
        let k = i;
        while (k < n && isWs(bytes[k])) k++;
        j = k === n ? k : k - i >= 2 ? k - 1 : k;
      }
      out.push([i, j]);
      i = j;
    }
    return out;
  }

  class RookTokenizer {
    constructor(spec) {
      if (!spec || spec.format !== "rook-bytebpe-v1") throw new Error("Unsupported tokenizer format");
      this.merges = spec.merges;
      const n = this.merges.length;
      this.n = n;
      this.pad = 256 + n;
      this.bos = 257 + n;
      this.eos = 258 + n;
      this.vocabSize = 259 + n;
      if (spec.vocab_size !== this.vocabSize || spec.pad !== this.pad || spec.bos !== this.bos || spec.eos !== this.eos) {
        throw new Error("Tokenizer special ids do not match its merge count");
      }
      this.rank = new Map();
      this.table = new Array(256 + n);
      for (let i = 0; i < 256; i++) this.table[i] = Uint8Array.of(i);
      for (let r = 0; r < n; r++) {
        const [a, b] = this.merges[r];
        this.rank.set(a * 65536 + b, r);
        const left = this.table[a];
        const right = this.table[b];
        const joined = new Uint8Array(left.length + right.length);
        joined.set(left, 0);
        joined.set(right, left.length);
        this.table[256 + r] = joined;
      }
      this.cache = new Map();
    }

    _encodePiece(bytes, start, end) {
      let key = "";
      for (let i = start; i < end; i++) key += String.fromCharCode(bytes[i]);
      const hit = this.cache.get(key);
      if (hit) return hit;
      let ids = [];
      for (let i = start; i < end; i++) ids.push(bytes[i]);
      while (ids.length > 1) {
        let bestRank = -1;
        let a = 0;
        let b = 0;
        for (let i = 0; i + 1 < ids.length; i++) {
          const r = this.rank.get(ids[i] * 65536 + ids[i + 1]);
          if (r !== undefined && (bestRank === -1 || r < bestRank)) {
            bestRank = r;
            a = ids[i];
            b = ids[i + 1];
          }
        }
        if (bestRank === -1) break;
        const newId = 256 + bestRank;
        const merged = [];
        for (let i = 0; i < ids.length; ) {
          if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) {
            merged.push(newId);
            i += 2;
          } else {
            merged.push(ids[i]);
            i += 1;
          }
        }
        ids = merged;
      }
      if (this.cache.size < 200000) this.cache.set(key, ids);
      return ids;
    }

    /** Encode a string (or Uint8Array of UTF-8 bytes) to token ids. Never emits special ids. */
    encode(input) {
      const bytes = typeof input === "string" ? new TextEncoder().encode(input) : input;
      const out = [];
      for (const [s, e] of pretokenize(bytes)) {
        const ids = this._encodePiece(bytes, s, e);
        for (let i = 0; i < ids.length; i++) out.push(ids[i]);
      }
      return out;
    }

    /** Bytes for a token id (empty for special ids). */
    tokenBytes(id) {
      return id < 256 + this.n ? this.table[id] : new Uint8Array(0);
    }

    decodeBytes(ids) {
      let total = 0;
      for (const id of ids) total += this.tokenBytes(id).length;
      const out = new Uint8Array(total);
      let at = 0;
      for (const id of ids) {
        const t = this.tokenBytes(id);
        out.set(t, at);
        at += t.length;
      }
      return out;
    }

    decode(ids) {
      return new TextDecoder("utf-8", { fatal: false }).decode(this.decodeBytes(ids));
    }
  }

  const api = { RookTokenizer, pretokenize };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RookTokenizerLib = api;
})(typeof self !== "undefined" ? self : this);
