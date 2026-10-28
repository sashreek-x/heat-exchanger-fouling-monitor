# Data directory

Do not commit confidential plant data or personal/operational secrets.

- `sample_schema.csv` documents column names and units; it is not a training dataset.
- `demo_synthetic.csv` is generated on demand by `python -m src.data.make_demo_data`. It is synthetic, illustrative data and must not be reported as real experimental evidence.
- For the final academic report, cite the real dataset source and its license, describe target construction and exclusions, and preserve a reproducible copy or download instructions where permitted.
