# Development Instructions

## Synthetic Data for Benchmarking and Testing

To generate synthetic data, first install `numpy` into a fresh Python virtual environment and then run the following command:

```sh
python scripts/dataset_generator.py --num_images 1000 --annotations_per_image 10 --output_dir ./assets
```

Feel free to tweak the parameters to generate more or less data.
