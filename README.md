Code Associated With: 
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
