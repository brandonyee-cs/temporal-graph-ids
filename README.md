Code Associated With: 

## T-GAT: Trustworthy Lateral Movement Detection via Temporal Graph Attention with Uncertainty Quantification

### Brandon Yee <sup>1</sup>

<sup>1</sup> Yee Collins Research Group, b.yee@ycrg-labs.org

---

### Installation

```bash
pip install -e .
```

### Data

Download the LANL dataset manually from https://csr.lanl.gov/data/cyber1/ and place files in `data/`.

### Training

```bash
python src/train.py --config configs/default.yaml
```

### Inference

```bash
python src/infer.py --model-path experiments/best_model.pt --auth-file data/auth.txt
```