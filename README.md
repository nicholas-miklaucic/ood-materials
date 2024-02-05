# OOD Materials Work

The code is in `src/`, configs are in `configs/`. Data is not bundled.

`dataset-normalization.ipynb` gets the materials data. `subsampling` shows how I make the splits for datasets.

To run the training loop with debugging output and things set so it only runs for a couple epochs, execute:

```bash
python src/train.py --smoke_test=True --cli.verbosity debug
```

`python src/train.py --help` to see all available options, or look at `src/config.py`.

Every `TODO` is something we need to look at.

Other things that are planned:
 - Config support for changing out datasets, using the splits I made
 - Replacing `Adam` with `AdamW`