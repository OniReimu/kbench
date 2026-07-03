# Datasheet: K-Bench Synthetic PII Corpus

Following Gebru et al., *Datasheets for Datasets*.

## Motivation

The corpus supplies controllable "private" facts whose location (substrate) can be
manipulated, so that unlearning can be evaluated without exposing any real person's
data. It exists because real-PII benchmarks cannot ethically vary *where* the secret
is injected (weights vs. context vs. retrieval).

## Composition

- **5,000 synthetic entities** (`pii-00000` … `pii-04999`), each with a canonical
  biography and five PII fields: date of birth, home address, occupation, employer,
  (plus email/phone in the bio text).
- **No real personal data.** All names, addresses, dates, employers are generated
  with [Python Faker](https://faker.readthedocs.io/). Any resemblance to a real
  person is coincidental.
- **Retrieval distractor corpus.** The retrieval substrate's document store
  (`data/wiki_index*`) ships real public-domain Wikipedia passages, used as a
  realistic distractor around the injected synthetic PII. Only the injected PII is
  synthetic; the surrounding corpus is intentionally real so retrieval behaves as it
  would in deployment.
- **Splits:** forget set = `pii-00000`…`pii-00999` (1,000); retain set =
  `pii-01000`…`pii-04999` (4,000). Each is further partitioned into disjoint
  adapter-training and evaluation pools (the "D7" split) to prevent in-sample
  inflation. A 250-entity holdout (seed 99, non-members) supports the membership
  -inference (PrivLeak / Min-K%) probe.
- **Real-PII-format check:** a separate block converts the public LUME benchmark's
  realistically-formatted records (forget-only, `date_of_birth`) as an external
  validity check; see the paper. LUME entities are *not* redistributed here — the
  conversion script (`scripts/19_convert_lume_to_kbench.py`) consumes LUME from its
  own source.

## Collection / generation

`scripts/01_generate_pii.py` (seeded Faker). Fully regenerable:

```bash
uv run python scripts/01_generate_pii.py --n-facts 5000 --seed 0 --out-dir data/pii_facts --name v1
```

## Uses

Intended: benchmarking unlearning / privacy-leakage evaluation. Not intended: any
use implying the entities are real, or training a model to associate the synthetic
names with the synthetic attributes outside this evaluation.

## Distribution & license

Released under CC BY 4.0. HuggingFace upload:

```bash
huggingface-cli upload <anon-org>/kbench-pii-corpus data/pii_facts --repo-type dataset
```

## Maintenance

Versioned with the code. Corpus regeneration is deterministic given the seed, so the
exact corpus is reproducible from `scripts/01_generate_pii.py` without redistributing
large files.
