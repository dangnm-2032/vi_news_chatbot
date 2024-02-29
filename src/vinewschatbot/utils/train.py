from unstructured.cleaners.core import clean_extra_whitespace

def clean_text(batch):
    batch['text'] = batch['text'].replace("\n", ' ')
    batch['text'] = batch['text'].replace("\t", ' ')
    batch['text'] = batch['text'].replace("\r", ' ')
    batch['text'] = clean_extra_whitespace(batch['text'])

    batch['summary'] = batch['summary'].replace("\n", ' ')
    batch['summary'] = batch['summary'].replace("\t", ' ')
    batch['summary'] = batch['summary'].replace("\r", ' ')
    batch['summary'] = clean_extra_whitespace(batch['summary'])

    return batch

def get_len(sample, tokenizer):
    sample['summary_len'] = len(tokenizer(sample['summary'])[0])
    sample['text_len'] = len(tokenizer(sample['text'])[0])
    return sample

def transform(batch, tokenizer):
    input = "vietnews: " + batch['text'] + " </s>"
    tokenized_input = tokenizer(
        input, 
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=896,
        add_special_tokens=False
    )
    batch['input_ids'] = tokenized_input['input_ids'][0]
    batch['attention_mask'] = tokenized_input['attention_mask'][0]

    tokenized_label = tokenizer(
        batch['summary'], 
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=129,
        add_special_tokens=False
    )
    batch['labels'] = tokenized_label['input_ids'][0]
    return batch