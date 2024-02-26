<!-- ABOUT THE PROJECT -->
## ObectDetection

![the picture](assets/READMEtitle.png)


<!-- Structure -->
## Structure
``` 
├── checkpoints: Directory containing information about different finetuning RUNS for each model architecture   
│   ├── yolo_nas_l   
│   │   ├── Runs named by mAP@0.50:0.95 on test set. E.g. 0417 = 41.7%   
│   │   └── ...   
│   ├── yolo_nas_s   
│       └── ...   
├── finetuning   
│   ├── config.json: config file that sets relevant parameters for finetuning   
│   ├── finetune.py: script to finetune YOLO NAS   
│   ├── split_dataset.py: script to create TRAIN-VAL-TEST split   
│   └── test.py: script to infer and save test metrics for single finetuning RUNS within checkpoints/architecture/   
├── od_train_data   
│   ├── dataset: annotated data in YOLO dataset format   
│   ├── dataset_stats: files that document the annotation distribution in the dataset   
│   └── raw_dataset: unsplitted, annotated data   
├── assets   
│   ├── analyze_dataset.py: script to output dataset distribution metrics for given YOLO dataset   
│   └── README_title.png: sample image for this README   
└── predict.py: script containing a PREDICTOR class that provides a predict funktion for our pipeline   
```

## Usage

**predict.py**    
Visualize finetuned model predictions for 3 random samples from the test set.   
```python predict.py```

**finetuning/split_dataset.py**    
Test a finetuning RUNs best wheights with the parameters set in config.json on the test set. Split retios and seed to be defined at top of script.    
```python split_dataset.py --datatset_folder 'path_to_ObjectDetection/od_train_data/raw_dataset' --output_folder 'path_to_ObjectDetection/od_train_data/<dataset_name>'```

**finetuning/finetune.py**    
Start a finetuning RUN with the parameters set in config.json    
```python finetune.py```

**finetuning/test.py**   
Test a finetuning RUNs best wheights with the parameters set in config.json on the test set    
```python test.py --architecture 'yolo_nas_l' --run '<name_of_RUN_folder>'```

**assets/predict.py**   
Visualize finetuned model predictions for 3 random samples from the test set.    
```python analyze_dataset.py --input_folder path_to_ObjectDetection/od_train_data/dataset/labels --output_folder path_to_ObjectDetection/od_train_data/dataset_stats```



