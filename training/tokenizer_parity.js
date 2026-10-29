// Used by test_tokenizer_js.py: encode the given cases with the browser tokenizer and print JSON.
// Usage: node tokenizer_parity.js <cases.json>
"use strict";
const fs = require("fs");
const path = require("path");
const { RookTokenizer, pretokenize } = require(path.join(__dirname, "..", "rook-tokenizer.js"));

const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const tok = new RookTokenizer(job.tokenizer);
const results = job.cases_hex.map((hex) => {
  const bytes = Uint8Array.from(Buffer.from(hex, "hex"));
  const ids = tok.encode(bytes);
  const pieces = pretokenize(bytes).map(([s, e]) => Buffer.from(bytes.subarray(s, e)).toString("hex"));
  const back = Buffer.from(tok.decodeBytes(ids)).toString("hex");
  return { ids, pieces, back };
});
process.stdout.write(JSON.stringify(results));
