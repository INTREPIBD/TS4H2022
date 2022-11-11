## Inferring mood disorder symptoms from multivariate time-series sensory data

Code for workshop paper "Inferring mood disorder symptoms from multivariate time-series sensory data" at the [NeurIPS 2022 Workshop on Learning from Time Series for Health](https://timeseriesforhealth.github.io/).

```
@unpublished{
    anonymous2022inferring,
    title={Inferring mood disorder symptoms from multivariate time-series sensory data},
    author={Anonymous},
    booktitle={NeurIPS 2022 Workshop on Learning from Time Series for Health},
    year={2022},
    url={https://openreview.net/forum?id=awjU8fCDZjS}
}
```

### Installation
- Create a new [conda](https://conda.io/en/latest/) environment with Python 3.8.
  ```bash
  conda create -n timebase python=3.8
  ```
- Activate `timebase` virtual environment
  ```bash
  conda activate timebase
  ```
- Install all dependencies and packages with `setup.sh` script, works on both Linus and macOS.
  ```bash
  sh setup.sh
  ```
  
### Dataset
- See [dataset/README.md](dataset/README.md) regarding data availability and the structure of the dataset.

### Train regression model
- To train a BiLSTM regression model with GRU embeddings
  ```
  python regression_train.py --output_dir runs/001_test_run --regression_mode 1 --qc_mode 1 --time_alignment 0 --embedding_type 0 --model bilstm --epochs 200 --verbose 1
  ```
- Use `python regression_train.py --help` to see all options.
- TensorBoard visualization
  ```
  tensorboard --logdir runs/001_test_run
  ```
