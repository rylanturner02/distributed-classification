# Distributed Tree Classifier

A simple distributed training system for tree classification using PyTorch. This architecture utilizes data parallelization, ideal for large-scale classification training.

## Setup

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Prepare your dataset with the following structure:

```
data/
├── train/
│   ├── oak/
│   │   ├── oak1.jpg
│   │   ├── oak2.jpg
│   │   └── ...
│   ├── pine/
│   │   ├── pine1.jpg
│   │   └── ...
│   └── maple/
│       ├── maple1.jpg
│       └── ...
└── val/
    ├── oak/
    ├── pine/
    └── maple/
```

3. Train the model:

```bash
python ./app/train.py --data_dir ./data
```

4. Start the prediction service:

```bash
python ./app/predict.py
```

5. Make predictions:

```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/predict
```

## System Features

- Distributed training using PyTorch DistributedDataParallel
- Simple FastAPI prediction service
- Proper error handling and logging
- Image preprocessing and data augmentation
