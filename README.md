# EvoHallu

Applying evolutionary optimization to improve hallucination detection.

# Setup

Create basic conda env:
```
cd EvoHallu
conda env create -f local_poly.yml 
conda activate local_poly
```
Install repo as a package, ensure you use `pip` prom conda environment:
```
which pip
pip install -e .
```
Add LM-Polygraph:
```
cd ..
git clone git@github.com:IINemo/lm-polygraph.git
cd lm-polygraph
pip install -e .
```