# Text and vision Transformers for image-to-recipe retrieval

Repository for image-to-recipe retrieval project for the DTU Course Advanced Deep Learning in Computer Vision (02501). You can find our poster [here](poster/poster_ADLCV.pdf)!

![alttext](reports/figures/architecture.png)

# Getting started
Create an environment with ``Python 3.10`` and install the dependencies.
```
pip install -r requirements.txt
```
# Train your own model


```
python src/models/train_model.py --image_size 224 --model_name "models/best_model_ever_1.pt" --mode 1 --pretrained
```

# Results
Image-to-text retrieval
![alt text](reports/figures/img2text.gif)

Text-to-image retrieval
![alt text](reports/figures/text2img.gif)