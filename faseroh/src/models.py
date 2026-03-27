# src/models.py
"""
Sequence-to-sequence models for symbolic Taylor expansion.

Classes
-------
LSTMSeq2Seq       – Bahdanau-attention bidirectional LSTM encoder-decoder
PositionalEncoding – sinusoidal PE used by the Transformer
TransformerSeq2Seq – standard Transformer encoder-decoder
"""

import math
import random
import torch
import torch.nn as nn

from .tokenizer import PAD, SOS, EOS


# --------------------------------------------------------------------------- #
#  LSTM seq2seq  (Bahdanau attention)
# --------------------------------------------------------------------------- #

class LSTMSeq2Seq(nn.Module):
    """
    Bidirectional LSTM encoder + LSTM decoder with Bahdanau attention.

    Plain single-vector LSTM bottleneck fails on symbolic math because a 30+
    character expression can't be compressed into one vector.  Attention lets
    the decoder inspect ALL encoder hidden states at each step.
    """

    def __init__(self, vsz: int, emb: int = 256, hid: int = 512):
        super().__init__()
        self.vsz      = vsz
        self.hid      = hid

        # encoder
        self.enc_emb  = nn.Embedding(vsz, emb, padding_idx=PAD)
        self.enc_rnn  = nn.LSTM(emb, hid, batch_first=True,
                                bidirectional=True, dropout=0.2)
        self.enc_proj = nn.Linear(hid * 2, hid)

        # decoder
        self.dec_emb  = nn.Embedding(vsz, emb, padding_idx=PAD)
        self.dec_rnn  = nn.LSTM(emb + hid, hid, batch_first=True, dropout=0.2)

        # attention
        self.attn     = nn.Linear(hid * 2, 1)

        # output projection
        self.proj     = nn.Linear(hid, vsz)

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _attend(self, dec_h, enc_out):
        """Bahdanau additive attention → context vector."""
        dec_h_exp = dec_h.repeat(1, enc_out.size(1), 1)          # (B, T_src, hid)
        scores    = self.attn(torch.tanh(
            torch.cat([dec_h_exp, enc_out], dim=-1)
        ))                                                         # (B, T_src, 1)
        weights   = torch.softmax(scores, dim=1)                  # (B, T_src, 1)
        return (weights * enc_out).sum(dim=1, keepdim=True)       # (B, 1, hid)

    def _encode(self, src):
        """Run the encoder and project to decoder hidden size."""
        enc_out, (h, _) = self.enc_rnn(self.enc_emb(src))        # bidir
        enc_out = self.enc_proj(enc_out)                          # (B, T, hid)
        h = torch.tanh(self.enc_proj(
            torch.cat([h[0], h[1]], dim=-1)                       # merge directions
        )).unsqueeze(0)                                            # (1, B, hid)
        return enc_out, h, torch.zeros_like(h)                    # enc_out, h0, c0

    # ------------------------------------------------------------------ #
    #  Forward (training)
    # ------------------------------------------------------------------ #

    def forward(self, src, tgt, tf: float = 0.5):
        """
        Teacher-forcing forward pass.

        Parameters
        ----------
        src : (B, T_src)
        tgt : (B, T_tgt)  — includes <sos> and <eos>
        tf  : teacher-forcing probability
        """
        enc_out, h, c = self._encode(src)
        B, T = tgt.shape
        outs = torch.zeros(B, T, self.vsz, device=src.device)
        tok  = tgt[:, 0]                                           # <sos>

        for step in range(1, T):
            emb      = self.dec_emb(tok.unsqueeze(1))             # (B, 1, emb)
            context  = self._attend(h.transpose(0, 1), enc_out)   # (B, 1, hid)
            dec_in   = torch.cat([emb, context], dim=-1)           # (B, 1, emb+hid)
            out, (h, c) = self.dec_rnn(dec_in, (h, c))
            logits   = self.proj(out.squeeze(1))                   # (B, vsz)
            outs[:, step] = logits
            tok = tgt[:, step] if random.random() < tf else logits.argmax(1)

        return outs

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, src, maxlen: int = 120):
        """Greedy decoding — fast, used during training eval."""
        self.eval()
        enc_out, h, c = self._encode(src)
        tok = torch.full((src.size(0),), SOS, dtype=torch.long, device=src.device)
        out = []
        for _ in range(maxlen):
            emb     = self.dec_emb(tok.unsqueeze(1))
            context = self._attend(h.transpose(0, 1), enc_out)
            e, (h, c) = self.dec_rnn(torch.cat([emb, context], dim=-1), (h, c))
            tok = self.proj(e.squeeze(1)).argmax(1)
            out.append(tok)
            if (tok == EOS).all():
                break
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def generate_beam(self, src, beam_width: int = 7, maxlen: int = 120):
        """
        Length-normalised beam search (batch size must be 1).
        Length penalty exponent = 0.6 (same as Google's GNMT paper).
        """
        self.eval()
        assert src.size(0) == 1, "beam search requires batch_size == 1"
        enc_out, h, c = self._encode(src)

        sequences = [([SOS], 0.0, h, c)]
        completed: list = []

        for _ in range(maxlen):
            all_cands = []
            for seq, score, h_, c_ in sequences:
                if seq[-1] == EOS:
                    completed.append((seq, score))
                    continue
                tok     = torch.tensor([[seq[-1]]], dtype=torch.long, device=src.device)
                emb     = self.dec_emb(tok)
                context = self._attend(h_.transpose(0, 1), enc_out)
                out, (h_new, c_new) = self.dec_rnn(
                    torch.cat([emb, context], dim=-1), (h_, c_)
                )
                lprobs = torch.log_softmax(self.proj(out.squeeze(1)), dim=-1)
                topk   = torch.topk(lprobs, beam_width)
                for j in range(beam_width):
                    tok_id    = topk.indices[0][j].item()
                    new_score = (score + topk.values[0][j].item()) / ((len(seq) + 1) ** 0.6)
                    all_cands.append((seq + [tok_id], new_score, h_new, c_new))

            if not all_cands:
                break
            sequences = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam_width]

        candidates = completed if completed else sequences
        best = max(candidates, key=lambda x: x[1])[0]
        return torch.tensor(best[1:], device=src.device).unsqueeze(0)


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d: int, drop: float = 0.1, maxlen: int = 512):
        super().__init__()
        self.drop = nn.Dropout(drop)

        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10_000.0) / d)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))               # (1, maxlen, d)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].to(x.device)
        return self.drop(x)


# --------------------------------------------------------------------------- #
#  Transformer seq2seq
# --------------------------------------------------------------------------- #

class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder Transformer for symbolic sequence transduction."""

    def __init__(
        self,
        vsz: int,
        d: int = 128,
        heads: int = 4,
        enc_layers: int = 3,
        dec_layers: int = 3,
        ff: int = 512,
        drop: float = 0.1,
    ):
        super().__init__()
        self.vsz = vsz
        self.d   = d

        self.src_emb     = nn.Embedding(vsz, d, padding_idx=PAD)
        self.tgt_emb     = nn.Embedding(vsz, d, padding_idx=PAD)
        self.dropout     = nn.Dropout(drop)
        self.pe          = PositionalEncoding(d, drop)
        self.transformer = nn.Transformer(
            d, heads, enc_layers, dec_layers, ff, drop, batch_first=True
        )
        self.proj        = nn.Linear(d, vsz)

    # ------------------------------------------------------------------ #
    #  Mask helpers
    # ------------------------------------------------------------------ #

    def _pad_mask(self, t):
        return t == PAD

    def _causal_mask(self, sz: int, dev):
        return torch.triu(torch.ones(sz, sz, device=dev), diagonal=1).bool()

    # ------------------------------------------------------------------ #
    #  Forward (training)
    # ------------------------------------------------------------------ #

    def forward(self, src, tgt):
        """
        Parameters
        ----------
        src : (B, T_src)
        tgt : (B, T_tgt - 1)   — right-shifted (no <eos> on input side)
        """
        scale = math.sqrt(self.d)
        se = self.dropout(self.pe(self.src_emb(src) * scale))
        te = self.dropout(self.pe(self.tgt_emb(tgt) * scale))
        out = self.transformer(
            se, te,
            tgt_mask                = self._causal_mask(tgt.size(1), src.device),
            src_key_padding_mask    = self._pad_mask(src),
            tgt_key_padding_mask    = self._pad_mask(tgt),
            memory_key_padding_mask = self._pad_mask(src),
        )
        return self.proj(out)                                      # (B, T_tgt-1, vsz)

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, src, maxlen: int = 120):
        """Autoregressive greedy decoding."""
        self.eval()
        scale = math.sqrt(self.d)
        s   = self.pe(self.src_emb(src) * scale)
        mem = self.transformer.encoder(s, src_key_padding_mask=self._pad_mask(src))
        ys  = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)
        for _ in range(maxlen):
            te  = self.pe(self.tgt_emb(ys) * scale)
            out = self.transformer.decoder(
                te, mem,
                tgt_mask              = self._causal_mask(ys.size(1), src.device),
                tgt_key_padding_mask  = self._pad_mask(ys),
            )
            nxt = self.proj(out[:, -1]).argmax(1, keepdim=True)
            ys  = torch.cat([ys, nxt], dim=1)
            if (nxt.squeeze(1) == EOS).all():
                break
        return ys[:, 1:]

    @torch.no_grad()
    def generate_beam(self, src, beam_width: int = 7, maxlen: int = 120):
        """Length-normalised beam search (batch size must be 1)."""
        self.eval()
        scale = math.sqrt(self.d)

        s   = self.pe(self.src_emb(src) * scale)
        mem = self.transformer.encoder(s, src_key_padding_mask=self._pad_mask(src))

        sequences = [([SOS], 0.0)]
        completed: list = []

        for _ in range(maxlen):
            all_cands = []
            for seq, score in sequences:
                if seq[-1] == EOS:
                    completed.append((seq, score))
                    continue
                ys     = torch.tensor(seq, device=src.device).unsqueeze(0)
                te     = self.pe(self.tgt_emb(ys) * scale)
                out    = self.transformer.decoder(
                    te, mem,
                    tgt_mask             = self._causal_mask(ys.size(1), src.device),
                    tgt_key_padding_mask = self._pad_mask(ys),
                )
                lprobs = torch.log_softmax(self.proj(out[:, -1]), dim=-1)
                topk   = torch.topk(lprobs, beam_width)
                for j in range(beam_width):
                    tok       = topk.indices[0][j].item()
                    new_score = (score + topk.values[0][j].item()) / ((len(seq) + 1) ** 0.6)
                    all_cands.append((seq + [tok], new_score))

            if not all_cands:
                break
            sequences = sorted(all_cands, key=lambda x: x[1], reverse=True)[:beam_width]

        candidates = completed if completed else sequences
        best = max(candidates, key=lambda x: x[1])[0]
        return torch.tensor(best[1:], device=src.device).unsqueeze(0)
