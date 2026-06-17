"""Integrity checks for the *fragmented*-SMILES views the model actually encodes.

Stage-2 SSL never feeds the model a plain SMILES — it feeds a ``fragment_view``:
a space-separated, shuffled subset of BRICS fragments (each carrying dummy-atom
tokens like ``[1*]``/``[16*]``), where the **space is the fragment separator**
(token id 4, ``frag_sep_id``). Two failure modes would silently corrupt training:

1. The 204-token char tokenizer has **no UNK**, so any out-of-vocab token is
   *dropped*, not flagged — a fragment could be silently mangled.
2. The fragment separator (" ") could be lost, collapsing distinct fragments.

These tests use genuinely complex, real fragment views from the dataset (not toy
SMILES) and assert the tokenization is lossless, the separators survive, the
fragments are valid molecules, and a forward pass yields sane embeddings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "artifacts" / "tokenizer" / "smiles_new.json"
DDIT_CKPT = REPO / "artifacts" / "checkpoints" / "invirtuo_gen.ckpt"
MOSES_SHARD = REPO / "artifacts" / "processed" / "moses" / "shard_0000.parquet"
BINDINGDB = REPO / "artifacts" / "processed" / "bindingdb" / "bindingdb_curated.parquet"

# The fragment separator the SSL module splits on (model/discrete_flow.yaml).
FRAG_SEP_ID = 4

requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER.is_file(), reason=f"tokenizer not found at {TOKENIZER}"
)
requires_shard = pytest.mark.skipif(
    not MOSES_SHARD.is_file(), reason=f"MOSES shard not found at {MOSES_SHARD}"
)
requires_ckpt = pytest.mark.skipif(
    not DDIT_CKPT.is_file(), reason=f"DDiT checkpoint not found at {DDIT_CKPT}"
)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER))


@lru_cache(maxsize=1)
def _complex_real_views() -> tuple[str, ...]:
    """The longest (most fragment-rich) real ``fragment_view`` strings in shard 0."""
    import pyarrow.parquet as pq

    from lattice_lab.preprocessing.molecules import fragment_view_column

    pf = pq.ParquetFile(str(MOSES_SHARD))
    col = fragment_view_column(pf.schema_arrow.names)
    batch = next(pf.iter_batches(batch_size=4000, columns=[col]))
    views = [str(v) for v in batch.column(col).to_pylist()]
    # Keep multi-fragment views, prefer the most complex (longest) ones.
    multi = [v for v in views if " " in v]
    multi.sort(key=len, reverse=True)
    chosen = tuple(multi[:8])
    assert chosen, "no multi-fragment views found in shard"
    return chosen


def _join_tokens(view: str) -> str:
    tok = _tokenizer()
    ids = tok.encode(view, add_special_tokens=False)
    return "".join(tok.convert_ids_to_tokens(ids))


# --------------------------------------------------------------------------- #
# Tokenizer / fragment integrity (no model — fast)
# --------------------------------------------------------------------------- #
@requires_tokenizer
def test_fragment_separator_encodes_to_frag_sep_id() -> None:
    """A single space must encode to exactly the id the SSL module splits on."""
    tok = _tokenizer()
    assert tok.encode(" ", add_special_tokens=False) == [FRAG_SEP_ID]


@requires_tokenizer
def test_tokenizer_has_no_unk_so_drops_are_silent() -> None:
    """Documents *why* we need the round-trip check: OOV tokens vanish silently."""
    assert _tokenizer().unk_token_id is None


@requires_tokenizer
@requires_shard
def test_real_fragment_views_tokenize_losslessly() -> None:
    """Encode→tokens→join must reproduce the view byte-for-byte (no silent drops)."""
    for view in _complex_real_views():
        assert _join_tokens(view) == view, f"lossy tokenization for: {view!r}"


@requires_tokenizer
@requires_shard
def test_fragment_separators_are_preserved() -> None:
    """Every inter-fragment space must survive as a frag_sep_id token."""
    tok = _tokenizer()
    for view in _complex_real_views():
        ids = tok.encode(view, add_special_tokens=False)
        n_sep_in = view.count(" ")
        n_sep_tok = sum(i == FRAG_SEP_ID for i in ids)
        assert n_sep_in >= 1
        assert n_sep_tok == n_sep_in, f"separators lost for: {view!r}"
        # n_fragments = separators + 1; splitting must agree.
        assert len(view.split(" ")) == n_sep_in + 1


@requires_tokenizer
@requires_shard
def test_training_shuffle_is_a_corruption_free_permutation() -> None:
    """The one training-specific op (``_two_views`` → ``shuffle_fragment_ids``)
    must be a pure token permutation: same tokenizer, same multiset of tokens,
    same number of fragment separators — so training can't see a corrupted
    molecule that inference/tests never would."""
    import random

    from lattice_lab.data.fragment_views import shuffle_fragment_ids

    tok = _tokenizer()
    rng = random.Random(0)
    for view in _complex_real_views():
        # Exactly the tokenization the SSL module's _two_views performs.
        body = tok.encode(view, add_special_tokens=False)
        shuffled = shuffle_fragment_ids(body, FRAG_SEP_ID, rng)
        assert sorted(shuffled) == sorted(body), "shuffle changed the token multiset"
        assert shuffled.count(FRAG_SEP_ID) == body.count(FRAG_SEP_ID)
        # Each resulting fragment still decodes to a valid molecule.
        from lattice_lab.data.fragment_views import split_fragment_ids

        for frag_ids in split_fragment_ids(shuffled, FRAG_SEP_ID):
            frag_smi = "".join(tok.convert_ids_to_tokens(frag_ids))
            if frag_smi:  # leading/trailing empties are allowed by the op
                from rdkit import Chem

                assert Chem.MolFromSmiles(frag_smi) is not None


@requires_shard
def test_real_view_fragments_are_valid_molecules() -> None:
    """Each whitespace-delimited fragment must parse with RDKit (dummies allowed)."""
    from rdkit import Chem

    for view in _complex_real_views():
        frags = view.split(" ")
        assert len(frags) >= 2
        for frag in frags:
            assert Chem.MolFromSmiles(frag) is not None, f"invalid fragment {frag!r}"


@requires_tokenizer
@pytest.mark.skipif(not BINDINGDB.is_file(), reason="bindingdb parquet not found")
def test_seeded_views_of_complex_molecules_are_intact() -> None:
    """The generation path (seeded_views) on complex real drugs round-trips too."""
    import pyarrow.parquet as pq
    from rdkit import Chem

    from lattice_lab.preprocessing.molecules import seeded_views

    pf = pq.ParquetFile(str(BINDINGDB))
    batch = next(pf.iter_batches(batch_size=4000, columns=["smiles"]))
    smiles = [str(s) for s in batch.column("smiles").to_pylist() if s]
    smiles.sort(key=len, reverse=True)  # the gnarliest molecules
    checked = 0
    for smi in smiles[:5]:
        views = seeded_views(smi, k=4)
        assert views, f"no views generated for {smi!r}"
        for view in views:
            for frag in view.split(" "):
                assert Chem.MolFromSmiles(frag) is not None
            assert _join_tokens(view) == view, f"lossy tokenization for {view!r}"
        checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# Forward pass on real fragment views (needs the DDiT checkpoint — slower)
# --------------------------------------------------------------------------- #
@requires_tokenizer
@requires_shard
@requires_ckpt
def test_encoder_forward_on_fragment_views_is_sane() -> None:
    """A real forward over complex fragment views yields finite, distinct,
    unit-norm, deterministic embeddings (no NaN/Inf from tokenization artifacts)."""
    import torch

    from lattice_lab.backbone.discrete_flow import build_discrete_flow_encoder

    enc = build_discrete_flow_encoder(
        ckpt_path=str(DDIT_CKPT),
        tokenizer_path=str(TOKENIZER),
        backbone_layer_start=6,
        backbone_layer_end=9,
        d_adapter=512,
        adapter_n_layers=4,
        encode_time=0.98,
        learnable_time=False,  # deterministic forward
        freeze_backbone=True,
        token_id_min=FRAG_SEP_ID,
        device="cpu",
    )
    enc.eval()

    views = list(_complex_real_views()[:5])
    with torch.no_grad():
        z = enc.encode_views(views, device="cpu")
        z2 = enc.encode_views(views, device="cpu")

    assert z.shape == (len(views), 512)
    assert torch.isfinite(z).all(), "non-finite values in z_m"
    # Adapter L2-normalizes z_m.
    assert torch.allclose(z.norm(dim=-1), torch.ones(len(views)), atol=1e-4)
    # Deterministic in eval mode.
    assert torch.allclose(z, z2, atol=1e-5)
    # Distinct molecules → distinct embeddings (not a collapsed representation).
    cos = z @ z.t()
    off_diag = cos[~torch.eye(len(views), dtype=torch.bool)]
    assert off_diag.max() < 0.999, "fragment views collapse to one embedding"
