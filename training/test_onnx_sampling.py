#!/usr/bin/env python3
"""Smoke-test real ONNX generation and reject obviously corrupted byte output.

This is a basic corruption guard, not a claim that a reply is semantically good.
"""
import re
from pathlib import Path

MODEL = Path("models/rook-step-105500.onnx")
session = None
CTX, PAD, BOS, EOS, VOCAB = 256, 256, 257, 258, 259


def generate(prompt_text, seed):
    global session
    import numpy as np
    if session is None:
        import onnxruntime as ort
        assert MODEL.is_file() and MODEL.stat().st_size > 50_000_000
        session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    prompt = [BOS] + list(("User: " + prompt_text + "\nAssistant:").encode("utf-8"))
    prompt = prompt[-CTX:]
    allow_unicode = bool(re.search(r"[\u0600-\u06FF]", bytes(prompt[1:]).decode("utf-8")))
    tokens = np.full((1, CTX), PAD, dtype=np.int64)
    tokens[0, :len(prompt)] = prompt
    rng = np.random.default_rng(seed)
    generated = []
    pending = 0
    lead = 0
    stop_reason = 'max_tokens'
    first_step_candidates = []
    for step in range(64):
        seq = (prompt + generated)[-CTX:]
        tokens.fill(PAD)
        tokens[0, :len(seq)] = seq
        logits = session.run(["logits"], {"tokens": tokens})[0][0, len(seq) - 1].astype(np.float64)
        logits[PAD] = -1e9
        logits[BOS] = -1e9
        if step < 8 or pending:
            logits[EOS] = -1e9
        # Match the browser's byte-level UTF-8 constraints so this test measures
        # model readability, not whether unconstrained sampling invents invalid UTF-8.
        for b in range(256):
            ok = allow_unicode or b < 127
            if pending:
                ok = 0x80 <= b <= 0xBF
                if pending == 2 and lead == 0xE0 and b < 0xA0: ok = False
                if pending == 2 and lead == 0xED and b > 0x9F: ok = False
                if pending == 3 and lead == 0xF0 and b < 0x90: ok = False
                if pending == 3 and lead == 0xF4 and b > 0x8F: ok = False
            else:
                if b < 32 and b not in (9, 10, 13): ok = False
                if 0x80 <= b <= 0xBF: ok = False
                if b in (0xC0, 0xC1) or b >= 0xF5: ok = False
            if not ok: logits[b] = -1e9
        top = np.argpartition(logits, -32)[-32:]
        if step == 0:
            first_step_candidates = sorted([(int(i), float(logits[i])) for i in top], key=lambda x: x[1], reverse=True)[:12]
        vals = logits[top]
        probs = np.exp((vals - vals.max()) / 0.72)
        probs /= probs.sum()
        nxt = int(rng.choice(top, p=probs))
        if nxt == EOS:
            stop_reason = 'EOS'
            break
        if not 0 <= nxt <= 255:
            stop_reason = f'invalid_special_token:{nxt}'
            break
        generated.append(nxt)
        if pending:
            pending -= 1
        elif 0xC2 <= nxt <= 0xDF:
            pending, lead = 1, nxt
        elif 0xE0 <= nxt <= 0xEF:
            pending, lead = 2, nxt
        elif 0xF0 <= nxt <= 0xF4:
            pending, lead = 3, nxt
        seq = (prompt + generated)[-CTX:]
        tokens.fill(PAD)
        tokens[0, :len(seq)] = seq
    if pending:
        generated = generated[:-pending]
    reply = bytes(generated).decode("utf-8", errors="strict")
    reply = "".join(ch for ch in reply if ch in "\n\t" or ord(ch) >= 32).strip()
    return reply, stop_reason, first_step_candidates, generated


def main():
    prompts = [
        "Hi.",
        "What is your name?",
        "How are you?",
        'What does "~" mean?',
        "What are you?",
        "What is 2 + 2?",
    ]
    relevance_cues = {
        "Hi.": {"hi", "hello", "hey", "greetings"},
        "What is your name?": {"rook", "name", "called"},
        "How are you?": {"fine", "good", "well", "okay", "ok", "doing", "feeling"},
        'What does "~" mean?': {"tilde", "approximately", "symbol"},
        "What are you?": {"rook", "model", "assistant", "program", "ai"},
        "What is 2 + 2?": {"4", "four"},
    }
    failed = False
    for index, prompt in enumerate(prompts):
        reply, stop_reason, candidates, token_ids = generate(prompt, 2026 + index)
        printable = sum(ch.isascii() and (ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"-()") for ch in reply)
        ratio = printable / max(1, len(reply))
        words = re.findall(r"[A-Za-z]{2,}", reply)
        normalized = set(re.findall(r"[a-z']+", reply.lower()))
        has_replacement = "\ufffd" in reply
        readable = len(reply) >= 8 and ratio >= 0.85 and len(words) >= 2 and not has_replacement
        relevant = bool(normalized & relevance_cues[prompt])
        passed = readable and relevant
        print(f"PROMPT: {prompt}")
        print(f"STOP_REASON: {stop_reason}; TOKEN_COUNT: {len(token_ids)}; FIRST_STEP_TOP: {candidates}")
        print(f"TOKEN_IDS: {token_ids[:40]}")
        print(f"REPLY: {reply!r}")
        print(f"PRINTABLE_ASCII_RATIO: {ratio:.3f}; WORDLIKE_TOKENS: {len(words)}; READABLE: {readable}; PROMPT_RELEVANT_CUE: {relevant}; PASS: {passed}")
        failed |= not passed

    if failed:
        raise SystemExit("Rook failed the basic readability or prompt-relevance guard. Do not publish this checkpoint as conversationally verified.")
    print("Basic readability and simple prompt-relevance guards passed. Human review is still required.")
    

if __name__ == "__main__":
    main()
