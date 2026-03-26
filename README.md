# FUZE – Fitting Utility for Quantum Optics & Cavity QED

**FUZE** je open-source Python nástroj pro přesné fitování a modelování dat z kvantové optiky, především z experimentů s defect centry (T-centra, G-centra atd.) v křemíku a fotonických kavitách.

Cílem je spojit ("fuse") experimentální data se teoretickými modely – rychle, reprodukovatelně a s automatickou detekcí, jestli v systému není nějaký skrytý extra decay kanál.

## Co FUZE umí

- **Stage A**: Fit transfer function na spektrální data (např. cavity transmission/reflection nebo emitter spectrum)
- **Ringdown analysis**: Fitování časových decay křivek (životnosti, Purcell enhancement)
- **Closure test**: Kontrola konzistence mezi různými měřeními (např. quantum efficiency, outcoupling)
- **Extra channel detection**: Používá Bayesian Information Criterion (BIC) k tomu, aby statisticky rozhodl, jestli data podporují přidání druhého (nebo více) nezávislého decay kanálu
- Generování testovacích dat pro validaci pipeline

Výstupy: tabulky parametrů, RMSE, BIC delta, JSON summary, ploty fitů a finální verdikt (True/False pro každou stage).

## Příklad použití

Na otevřeném datasetu z článku  
**Cavity-coupled telecom atomic source in silicon** (Nature Communications, 2024)  
(https://doi.org/10.1038/s41467-024-46643-8)

```bash
python fuze_mat.py --file fig2c_data.mat   --mat-key 'Sample A, 5$\mathcal{\times10^{13} cm^{-2}}$'   --output-dir D:\hit\PythonProject\   --max-peaks 3   --max-baseline-degree 2

python make_stage_b_test_data.py

python HOCUS_POKUS_3.py --core-summary core_summary.json --spectrum fig2c_data.mat --dataset-key "Sample A, 5$\mathcal{\times10^{13} cm^{-2}}$" --ringdown ringdown.csv --ringdown-xcol t --ringdown-ycol alpha --closure closure.csv --closure-xcol t --closure-ycol T --out-prefix final
```
