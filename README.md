# DPP contracting game

Code and data for the paper:

**Contracting for Digital Product Passports: A Game-Theoretic Analysis of Supplier Data Sharing in Circular Supply Chains**

The scripts define the buyer-supplier game, derive equilibrium conditions, run the simulations, and rebuild the tables and figures used in the manuscript.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt

python code/02_model_setup.py
python code/03_equilibrium_analysis.py
python code/04_simulation.py
python code/05_make_tables_figures.py
```

The literature search script is separate because it calls external scholarly APIs:

```bash
python code/01_search_literature.py
```
