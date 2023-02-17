# Fairness without Sensitive Information: A Class-Wise Weighted Risk

This repo contains pytorch implementation of "Fairness without Sensitive Information: A Class-Wise Weighted Risk
Minimization Approach"

## Prerequisties

Refer to `requirements.txt`

## Training Details

- You can train AW-ERM on `imsitu` dataset with
````
sh train.sh
````

Then model will be saved in `models` directory and log files in `logs` directory.


## Download dataset

To simulate the experiment, you can download the dataset
- [imsitu](https://drive.google.com/file/d/1I9nIAh9bUuOtLIAnvzdZ9nKTROTDn9eA/view?usp=share_link)

Add the unzipped data in to `verb_classification` directory.