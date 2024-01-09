# vi_news_chatbot

0. Choose encode model
    a. Vietnamese dataset
    b. Sentence similarity task
    c. News dataset
    -> bkai-foundation-models/vietnamese-bi-encoder

1. Crawl dataset
    0. HF Datasets
        - vietgpt/news_summarization_vi
        - OpenHust/vietnamese-summarization
        - nlplabtdtu/summarization_sft_prompted
        - csebuetnlp/xlsum
        - toanduc/t5-sumary-dataset
        - bltlab/lr-sum
    -> Done
    a. VNExpress
    b. Crawl format
        {
            'unix_timestamp': '',
            'title': '',
            'content': '',
            'images': {
                '0': {
                    'caption': '',
                    'img_path': ''
                }
            }
        }
2. Database
    a. Tables: 
        - News (id, title) 
        - Passage (id, news_id, data)
        - Chunk (id, passage_id, data, embedding)

3. Train summarize model
    a. Dataset
        - Query ~10 related news, concat them into one doc -> input
        - Use LLM to summarize it -> label
    b. Model
        - NlpHUST/t5-small-vi-summarization
        - VietAI/vit5-large-vietnews-summarization
        - PhoGPT
        - bkai-foundation-models/vietnamese-llama2-7b-120GB
        