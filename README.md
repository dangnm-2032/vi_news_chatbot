# vi_news_chatbot

0. Choose encode model
    a. Vietnamese dataset
    b. Sentence similarity task
    c. News dataset

1. Crawl dataset
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