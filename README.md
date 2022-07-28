# CleanedCriticGuided

### Train Critic based on cls token
python main.py -train --model FOLDER_MODEL -vits

### Visualize attention
python VisualiuzeAttention.py --image_path [imagePath]

### CLS manipulation
#### Training with bins
python mainTrainingBins.py -train --model FOLDER_MODEL -usebins --pretrained_weights [dino_weights]

#### Swap Cls
python ManipulateCLS.py --pretrained_weights [dino_weights] --image_path1 [img_path] --image_path2 [img_path]


### Automated cell selection
python main_automatedSelection.py -kmean #pick Cells

python main_automatedSelection.py -oneep # vis for one ep + second clustering

python evaluation.py -test --model FOLDER_MODEL --output-video FOLDER_VIDEO  # evaluation

### Upper-bound
python evaluation.py -test --model FOLDER_MODEL --output-video FOLDER_VIDEO -count

python evaluation.py -test --model FOLDER_MODEL --output-video FOLDER_VIDEO -compute

### Discriminator
python evaluation.py -test --model FOLDER_MODEL --output-video FOLDER_VIDEO -discrim

Sources:

https://github.com/ndrwmlnk/critic-guided-segmentation-of-rewarding-objects-in-first-person-views

https://github.com/facebookresearch/dino