# CS7650-project-vqa
#### Images: Training images (5k), Validation images (1k)
https://drive.google.com/open?id=1cLUCEGM4UW_GrI5iZdZfdLaPjM0hYvx_
<br><br>
#### Pre-processed data:
###### answer_tokens_top1k.json: Contains the Top 1k answers (sorted by frequency) from the training dataset in tokenised form.
###### train_dataset_5k.json: Contains 5k randomly chosen question, answer, image triplets from the training dataset.
###### val_dataset_1k.json: Contains 1k randomly chosen question, answer, image triplets from the validation dataset.

###### train_dataset_5k_tokenised.json: Contains most common answer (amongst 10 answers for each question with count >= 3) in tokenised form using tokenisation dictionary from answer_tokens_top1k.json
###### val_dataset_1k_tokenised.json: Contains most common answer (amongst 10 answers for each question with count >= 3) in tokenised form using tokenisation dictionary from answer_tokens_top1k.json
