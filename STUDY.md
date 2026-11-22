# ConTEB experiment protocol

Run these commands in PowerShell from the repository root with the project Python environment active. Use a new `$Run` directory for each experiment. The export command uses the optional `datasets` package and downloads the published data. It records the resolved upstream commit in each manifest. Evaluation consumes the exported files locally.

## Prepare native chunks and relevance labels

```powershell
python -m pip install datasets
$Run = 'results/conteb-v1'
python scripts/prepare_conteb.py --repo illuin-conteb/mldr-conteb-train --source-split train --dataset conteb-mldr --split train --out "$Run/mldr-train.json"
python scripts/prepare_conteb.py --repo illuin-conteb/mldr-conteb-eval --source-split test --dataset conteb-mldr --split test --out "$Run/mldr-test.json"
python scripts/prepare_conteb.py --repo illuin-conteb/covid-qa --source-split train --dataset conteb-covid-qa --split test --out "$Run/covid-qa.json"
python scripts/prepare_conteb.py --repo illuin-conteb/insurance --source-split train --dataset conteb-insurance --split test --out "$Run/insurance.json"
```

COVID-QA and Insurance publish their evaluation data under the upstream split name `train`. Here `test` designates their study role. The exporter reads the published `documents` and `queries` configurations. The loader sorts native chunks by their numeric suffix and preserves chunk text and IDs. When title metadata is absent, Independent + Prefix uses up to 64 tokens from the lead chunk. The target span is pooled even when its text is also present in the prefix.

Published datasets: [MLDR training](https://huggingface.co/datasets/illuin-conteb/mldr-conteb-train), [MLDR evaluation](https://huggingface.co/datasets/illuin-conteb/mldr-conteb-eval), [COVID-QA](https://huggingface.co/datasets/illuin-conteb/covid-qa), [Insurance](https://huggingface.co/datasets/illuin-conteb/insurance).

## Select settings on MLDR train and freeze them

The example sweeps `k ∈ {4, 8}` for anchor frequency and overlap of 12.5% or 25% of the named cap. Keep the same encoder revision, runtime settings, and eligibility across the sweep. The example uses the Jina revision exercised in integration.

```powershell
$ModelRevision = '44e7d1d6caec8c883c2d4b207588504d519788d0'
foreach ($Cap in 1024, 2048, 4096) {
    foreach ($Overlap in ($Cap / 8), ($Cap / 4)) {
        foreach ($K in 4, 8) {
            $Trial = "cap$Cap-overlap$Overlap-k$K"
            python scripts/run_study.py --native "$Run/mldr-train.json" --phase development --context-cap $Cap --overlap $Overlap --anchor-k $K --model-revision $ModelRevision --tokenizer-revision $ModelRevision --out "$Run/dev/$Trial.json" --write-frozen-config "$Run/dev/$Trial.frozen.json"
            if ($LASTEXITCODE) { throw "Development run failed for $Trial" }
        }
    }
}
```

Record which configuration was selected for each context cap and the development criterion before inspecting held-out metrics. The dev result and its frozen configuration share the same trial name. Use edit runs on the development set if refresh cost informs selection. `--limit-docs` in the edit runner counts sampled documents. Record that sample and the selection rule separately from the full quality set.

## Evaluate the frozen selection

Set `$Trial` to a configuration selected on development data, then repeat for the other selected caps. The held-out quality run emits per-query rankings, nDCG@10, Recall@10, explicit coverage, and a source-document paired bootstrap for Content-Anchored minus Long Late. The proposal's noninferiority threshold is a 95% confidence interval lower bound greater than −0.02 nDCG. A dataset with one source document supports descriptive results only.

```powershell
$Frozen = "$Run/dev/$Trial.frozen.json"
python scripts/run_study.py --native "$Run/mldr-test.json" --phase held-out --frozen-config $Frozen --out "$Run/test/$Trial.json"
python scripts/run_study_edits.py --native "$Run/mldr-test.json" --frozen-config $Frozen --limit-docs 20 --edit-position all --out "$Run/edits/$Trial.json"
python scripts/run_study.py --native "$Run/covid-qa.json" --phase transfer --frozen-config $Frozen --out "$Run/transfer/covid-qa-$Trial.json"
python scripts/run_study.py --native "$Run/insurance.json" --phase transfer --frozen-config $Frozen --out "$Run/transfer/insurance-$Trial.json"
```

The edit run measures exact inference reuse, rebuilt-vector agreement, and latency for synthetic equal-length replacements, insertions, deletions, and prefix changes. Audited labels would be required to measure retrieval quality after content edits. Full Late eligibility is applied to the corpus and edit snapshots across all policies. A chunk beyond the matched cap can run alone if it fits the model. The wrapped Long Late windows restart on either side of such a singleton, while their ordinary overlap ownership follows Algorithm 2. For latency comparisons, repeat across runs on the same device with a fixed software environment and report the sample size and timing variation.
