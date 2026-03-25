"""models.py — LSTM seq2seq and Transformer seq2seq architectures."""

import math
import random

import torch
import torch.nn as nn


# ── LSTM Seq2Seq with Bahdanau Attention ──────────────────────────────────────

class LSTMSeq2Seq(nn.Module):
    """
    Attention-based LSTM encoder-decoder.

    Plain LSTM (single h,c bottleneck) fails on symbolic math —
    the encoder cannot compress a 30+ char expression into one vector.
    Bahdanau attention lets the decoder attend ALL encoder hidden states.
    """

    def __init__(self, vsz: int, emb: int = 256, hid: int = 512, pad_idx: int = 0):
        super().__init__()
        self.vsz      = vsz
        self.hid      = hid
        self.pad_idx  = pad_idx

        self.enc_emb  = nn.Embedding(vsz, emb, padding_idx=pad_idx)
        self.enc_rnn  = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(hid * 2, hid)
        self.dec_emb  = nn.Embedding(vsz, emb, padding_idx=pad_idx)
        self.dec_rnn  = nn.LSTM(emb + hid, hid, batch_first=True)
        self.attn     = nn.Linear(hid * 2, 1)
        self.proj     = nn.Linear(hid, vsz)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _attend(self, dec_h, enc_out):
        scores  = self.attn(torch.cat([dec_h.expand_as(enc_out), enc_out], dim=-1))
        weights = torch.softmax(scores, dim=1)
        return (weights * enc_out).sum(dim=1, keepdim=True)

    def _encode(self, src):
        enc_out, (h, _) = self.enc_rnn(self.enc_emb(src))
        enc_out = self.enc_proj(enc_out)
        h = torch.tanh(
            self.enc_proj(torch.cat([h[0], h[1]], dim=-1))
        ).unsqueeze(0)
        return enc_out, h, torch.zeros_like(h)

    # ── forward / generate ───────────────────────────────────────────────────

    def forward(self, src, tgt, tf: float = 0.5):
        enc_out, h, c = self._encode(src)
        B, T = tgt.shape
        outs = torch.zeros(B, T, self.vsz, device=src.device)
        tok  = tgt[:, 0]
        for step in range(1, T):
            emb       = self.dec_emb(tok.unsqueeze(1))
            context   = self._attend(h.transpose(0, 1), enc_out)
            out, (h, c) = self.dec_rnn(torch.cat([emb, context], dim=-1), (h, c))
            logits       = self.proj(out.squeeze(1))
            outs[:, step] = logits
            tok = tgt[:, step] if random.random() < tf else logits.argmax(1)
        return outs

    @torch.no_grad()
    def generate(self, src, eos_id: int = 2, maxlen: int = 120):
        """Greedy decoding."""
        self.eval()
        enc_out, h, c = self._encode(src)
        tok = torch.full((src.size(0),), 1, dtype=torch.long, device=src.device)  # SOS=1
        out = []
        for _ in range(maxlen):
            emb     = self.dec_emb(tok.unsqueeze(1))
            context = self._attend(h.transpose(0, 1), enc_out)
            e, (h, c) = self.dec_rnn(torch.cat([emb, context], dim=-1), (h, c))
            tok = self.proj(e.squeeze(1)).argmax(1)
            out.append(tok)
            if (tok == eos_id).all():
                break
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def generate_beam(self, src, sos_id: int = 1, eos_id: int = 2,
                      beam_width: int = 5, maxlen: int = 120):
        """Length-normalised beam search (batch_size must be 1)."""
        self.eval()
        assert src.size(0) == 1, "beam search: batch size must be 1"
        enc_out, h, c = self._encode(src)
        sequences = [([sos_id], 0.0, h, c)]
        completed = []

        for _ in range(maxlen):
            all_cands = []
            for seq, score, h_, c_ in sequences:
                if seq[-1] == eos_id:
                    completed.append((seq, score))
                    continue
                tok     = torch.tensor([[seq[-1]]], dtype=torch.long, device=src.device)
                emb     = self.dec_emb(tok)
                context = self._attend(h_.transpose(0, 1), enc_out)
                out, (hn, cn) = self.dec_rnn(torch.cat([emb, context], dim=-1), (h_, c_))
                lprobs  = torch.log_softmax(self.proj(out.squeeze(1)), dim=-1)
                topk    = torch.topk(lprobs, beam_width)
                for j in range(beam_width):
                    tok_id    = topk.indices[0][j].item()
                    new_score = (score + topk.values[0][j].item()) / ((len(seq) + 1) ** 0.7)
                    all_cands.append((seq + [tok_id], new_score, hn, cn))
            if not all_cands:
                break
            sequences = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam_width]

        best = (sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
                if completed else sequences[0][0])
        return torch.tensor(best[1:], device=src.device).unsqueeze(0)


# ── Transformer Seq2Seq ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d: int, drop: float = 0.1, maxlen: int = 512):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vsz: int, d: int = 128, heads: int = 4,
                 enc_layers: int = 3, dec_layers: int = 3,
                 ff: int = 512, drop: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vsz     = vsz
        self.d       = d
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(vsz, d, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(vsz, d, padding_idx=pad_idx)
        self.pe      = PositionalEncoding(d, drop)
        self.transformer = nn.Transformer(
            d, heads, enc_layers, dec_layers, ff, drop, batch_first=True
        )
        self.proj = nn.Linear(d, vsz)

    def _pad_mask(self, t):
        return t == self.pad_idx

    def _causal_mask(self, sz, dev):
        return torch.triu(torch.ones(sz, sz, device=dev), diagonal=1).bool()

    def forward(self, src, tgt):
        scale = math.sqrt(self.d)
        se = self.pe(self.src_emb(src) * scale)
        te = self.pe(self.tgt_emb(tgt) * scale)
        out = self.transformer(
            se, te,
            tgt_mask               = self._causal_mask(tgt.size(1), src.device),
            src_key_padding_mask   = self._pad_mask(src),
            tgt_key_padding_mask   = self._pad_mask(tgt),
            memory_key_padding_mask= self._pad_mask(src),
        )
        return self.proj(out)

    @torch.no_grad()
    def generate(self, src, sos_id: int = 1, eos_id: int = 2, maxlen: int = 120):
        """Greedy decoding."""
        self.eval()
        scale = math.sqrt(self.d)
        s   = self.pe(self.src_emb(src) * scale)
        mem = self.transformer.encoder(s, src_key_padding_mask=self._pad_mask(src))
        ys  = torch.full((src.size(0), 1), sos_id, dtype=torch.long, device=src.device)
        for _ in range(maxlen):
            te  = self.pe(self.tgt_emb(ys) * scale)
            out = self.transformer.decoder(
                te, mem, tgt_mask=self._causal_mask(ys.size(1), src.device)
            )
            nxt = self.proj(out[:, -1]).argmax(1, keepdim=True)
            ys  = torch.cat([ys, nxt], dim=1)
            if (nxt.squeeze(1) == eos_id).all():
                break
        return ys[:, 1:]

    @torch.no_grad()
    def generate_beam(self, src, sos_id: int = 1, eos_id: int = 2,
                      beam_width: int = 5, maxlen: int = 120):
        """Length-normalised beam search (batch_size must be 1)."""
        self.eval()
        scale = math.sqrt(self.d)
        s   = self.pe(self.src_emb(src) * scale)
        mem = self.transformer.encoder(s, src_key_padding_mask=self._pad_mask(src))

        sequences = [([sos_id], 0.0)]
        completed = []

        for _ in range(maxlen):
            all_cands = []
            for seq, score in sequences:
                if seq[-1] == eos_id:
                    completed.append((seq, score))
                    continue
                ys     = torch.tensor(seq, device=src.device).unsqueeze(0)
                te     = self.pe(self.tgt_emb(ys) * scale)
                out    = self.transformer.decoder(
                    te, mem, tgt_mask=self._causal_mask(ys.size(1), src.device)
                )
                lprobs = torch.log_softmax(self.proj(out[:, -1]), dim=-1)
                topk   = torch.topk(lprobs, beam_width)
                for j in range(beam_width):
                    tok_id    = topk.indices[0][j].item()
                    new_score = (score + topk.values[0][j].item()) / ((len(seq) + 1) ** 0.7)
                    all_cands.append((seq + [tok_id], new_score))
            if not all_cands:
                break
            sequences = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam_width]

        best = (sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
                if completed else sequences[0][0])
        return torch.tensor(best[1:], device=src.device).unsqueeze(0)
