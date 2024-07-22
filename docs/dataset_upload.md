# How to Add a Dataset to OpenQDC

Do you think that OpenQDC is missing some important dataset? Do you think your dataset would be a good fit for OpenQDC?
If so, you can contribute to OpenQDC by adding your dataset to the OpenQDC repository in two ways:

1. Opening a PR to add a new dataset
2. Request a new dataset through Google Form

## OpenQDC PR Guidelines

Implement your dataset in the OpenQDC repository by following the guidelines below:

### Dataset class

- The dataset class should be implemented in the `openqdc/datasets` directory.
- The dataset class should inherit from the `openqdc.datasets.base.BaseDataset` class.
- Add your `dataset.py` file to the `openqdc/datasets/potential` or `openqdc/datasets/interaction/` directory based on the type of energy.
- Implement the following for your dataset:
  - Add the metadata of the dataset:
    - Docstrings for the dataset class. Docstrings should report links and references to the dataset. A small description and if possible, the sampling strategy used to generate the dataset.
    - `__links__`: Dictionary of name and link to download the dataset.
    - `__name__`: Name of the dataset. This will create a folder with the name of the dataset in the cache directory.
    - The original units for the dataset `__energy_unit__` and `__distance_unit__`.
    - `__force_mask__`: Boolean to indicate if the dataset has forces. Or if multiple forces are present. A list of booleans.
    - `__energy_methods__`: List of the `QmMethod` methods present in the dataset.
  - `read_raw_entries(self)` -> `List[Dict[str, Any]]`: Preprocess the raw dataset and return a list of dictionaries containing the data. For a better overview of the data format. Look at data storage. This data should have the following keys:
    - `atomic_inputs` : Atomic inputs of the molecule. numpy.Float32.
    - `name`: Atomic numbers of the atoms in the molecule. numpy.Object.
    - `subset`: Positions of the atoms in the molecule.  numpy.Object.
    - `energies`: Energies of the molecule. numpy.Float64.
    - `n_atoms`: Number of atoms in the molecule. numpy.Int32
    - `forces`: Forces of the molecule. [Optional] numpy.Float32.
  - Add the dataset import to the `openqdc/datasets/<type_of_dataset>/__init__.py` file and to `openqdc/__init__.py`.

### Test the dataset

Try to run the openQDC CLI pipeline with the dataset you implemented.

Run the following command to download the dataset:

- Fetch the dataset files
```bash
openqdc fetch DATASET_NAME
```

- Preprocess the dataset
```bash
openqdc preprocess DATASET_NAME
```

- Load it on python and check if the dataset is correctly loaded.
```python
from openqdc import DATASET_NAME
ds=DATASET_NAME()
```

If the dataset is correctly loaded, you can open a PR to add the dataset to OpenQDC.

- Select for your PR the `dataset` label.

Our team will review your PR and provide feedback if necessary. If everything is correct, your dataset will be added to OpenQDC remote storage.

## OpenQDC Google Form

Alternatively, you can ask the OpenQDC main development team to take care of the dataset upload for you.
You can fill out the Google Form [here](https://docs.google.com/forms/d/e/1FAIpQLSeh0YHRn-OoqPpUbrL7G-EOu3LtZC24rtQWwbjJaZ-2V8P2vQ/viewform?usp=sf_link)

As the openQDC team will strive to provide a high quality curation and upload,
please be patient as the team will need to review the dataset and carry out the necessary steps to ensure the dataset is uploaded correctly.
