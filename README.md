## Integrating Text guided Multimodal attention for Image Super Resolution
This work is the end-semester term project for MM-805. 

### Download the dataset
Please download DIV2K dataset from [https://data.vision.ee.ethz.ch/cvl/DIV2K/]

### Creating Textual Descriptions
Run all cells of `create_desc.ipynb`. Note: Please change paths to your respective directories.

### Creating Textual Embeddings
Run all cells of `create_text_embed.ipynb`. Note: Please change paths to your respective directories.

### Train model
Run `python train.py` 

### Visualize results on Tensorboard
Run `tensorboard --logdir /runs --port 6006`
To visualize exiting tf events, check out runs directory
