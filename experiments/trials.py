"""
    0. Choose encode model
        a. Vietnamese dataset
        b. Sentence similarity task
        c. News dataset

    1. Crawl news from vnexpress
        a. Crawl format
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

"""
from bs4 import BeautifulSoup
import requests
import datetime
import time
from datasets import Dataset, disable_caching, load_from_disk, load_dataset, concatenate_datasets
from unstructured.cleaners.core import clean_extra_whitespace
from transformers import AutoConfig, T5ForConditionalGeneration, AutoTokenizer, GenerationConfig
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding
import torch
import gradio as gr
from underthesea import sent_tokenize
import pandas as pd
import numpy as np
import wandb
import gradio
import re
import string
disable_caching()

def crawl_post_link():
    # Constants
    CATEIDS = {
        'thoisu': 1001005,
        'thegioi': 1001002,
        'kinhdoanh': 1003159,
        'batdongsan': 1005628,
        'khoahoc': 1001009,
        'giaitri': 1002691,
        'phapluat': 1001007,
        'giaoduc': 1003497,
        'suckhoe': 1003750,
        'doisong': 1002966,
        'dulich': 1003231,
        'sohoa': 1002592,
        'xe': 1001006,
    }
    ONEDAY_TS = 86400
    vne_search_url = 'https://vnexpress.net/category/day/cateid/{cateid}/fromdate/{fromdate}/todate/{todate}/allcate/{cateid}'

    # Get today datetime
    today_datetime = datetime.datetime.utcnow()
    today_ts = int(time.mktime(today_datetime.utctimetuple()))
    print(f"Today is {today_datetime}\nUnix timestamp is {today_ts}")
    current_ts = today_ts
    crawl_data = []
    loop = True
    while loop:
        # Get yesterday timestamp
        print(f"Start crawling from {current_ts} <-> {datetime.datetime.utcfromtimestamp(current_ts).strftime('%Y-%m-%d')}")
        start = time.time()
        count = 0
        for i, catename in enumerate(CATEIDS):
            print(f"Category - {i+1}/{len(CATEIDS)}", end='\r')
            cateid = CATEIDS[catename]
            url = vne_search_url.format(cateid=cateid, fromdate=current_ts, todate=current_ts)
            while True:
                response = requests.get(url)
                if response.ok:
                    break
                time.sleep(0.5)
            soup = BeautifulSoup(response.content, "html.parser")
            links = []
            for article in soup.find_all("div", attrs="list-news-subfolder")[0].find_all("article"):
                links.append(article.h3.a.get("href"))
            
            if not links:
                loop = False
            count += len(links)
            for link in links:
                if link.find('.vnexpress') == -1:
                    crawl_data.append(
                        {
                            "unix_timestamp": current_ts,
                            "category_id": cateid,
                            "article_links": link
                        }
                    )
        try:
            del links, soup, article, url, response, cateid, catename
        except:
            pass
        end = time.time()
        print(f"Done - Time taken: {end - start} - Articles: {count}")
        current_ts -= ONEDAY_TS
        
    crawl_data = Dataset.from_list(crawl_data)
    print(crawl_data)
    crawl_data.save_to_disk("artifacts/post_link_data")
    del today_datetime, today_ts, yesterday_ts, crawl_data, start, count
    return

def crawl_post_content():
    def transform(sample):
        delay = 0.0
        url = sample['article_links']
        try:
            while True:
                response = requests.get(url)
                if response.ok:
                    break
                time.sleep(0.5 + delay)
                delay += 0.1
                if 0.5 + delay > 1.0:
                    print("Time out")
                    raise Exception("Time out")
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.text
            article_0 = []
            article_1 = []
            for p in soup.find_all("body")[0].find_all("p"):
                if p.text not in article_0 and not p.find("strong"):
                    article_0.append(p.text)
            for p in soup.find_all("p", "Normal"):
                if p.text not in article_1 and not p.find("strong"):
                    article_1.append(p.text)
            article = article_0 if len(article_0) >= len(article_1) else article_1
            content = " ".join(article)
            sample['title'] = title
            sample['content'] = content
        except Exception as e:
            print(e)
            sample['title'] = ""
            sample['content'] = ""
        return sample
    data_link_path = "/home/yuuhanase/FPTU/DAT/vi_news_chatbot/artifacts/post_link_data"
    crawl_data = load_from_disk(data_link_path)
    content_data = crawl_data.map(transform, batched=False)
    content_data.save_to_disk("/home/yuuhanase/FPTU/DAT/vi_news_chatbot/artifacts/post_content_data")
    del data_link_path, crawl_data, content_data
    return

def create_dataset():
    bltlab_ds = load_dataset('bltlab/lr-sum', 'vie')
    toanduc_ds = load_dataset('toanduc/t5-sumary-dataset')
    csebuetnlp_ds = load_dataset('csebuetnlp/xlsum', 'vietnamese')
    nlplabtdtu_ds = load_dataset('nlplabtdtu/summarization_sft_prompted')
    vietgpt_ds = load_dataset('vietgpt/news_summarization_vi')

    bltlab_full_ds = concatenate_datasets([bltlab_ds['train'], bltlab_ds['test'], bltlab_ds['validation']])
    toanduc_full_ds = concatenate_datasets([toanduc_ds['train']])
    csebuetnlp_full_ds = concatenate_datasets([csebuetnlp_ds['train'], csebuetnlp_ds['test'], csebuetnlp_ds['validation']])
    nlplabtdtu_full_ds = concatenate_datasets([nlplabtdtu_ds['train'], nlplabtdtu_ds['test']])
    vietgpt_full_ds = concatenate_datasets([vietgpt_ds['train'], vietgpt_ds['test']])

    bltlab_full_ds = bltlab_full_ds.remove_columns(['id', 'url', 'title'])
    toanduc_full_ds = toanduc_full_ds.rename_column('Original Text', 'text').rename_column('Summary', 'summary')
    csebuetnlp_full_ds = csebuetnlp_full_ds.remove_columns(['id', 'url', 'title'])
    nlplabtdtu_full_ds = nlplabtdtu_full_ds
    vietgpt_full_ds = vietgpt_full_ds.rename_column('content', 'text')

    full_ds = concatenate_datasets([bltlab_full_ds, toanduc_full_ds, csebuetnlp_full_ds, nlplabtdtu_full_ds, vietgpt_full_ds])
    full_ds = full_ds.shuffle(seed=42)
    full_ds.save_to_disk('artifacts/summary_dataset')

def train():
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Vi_news_chatbot",
        # Track hyperparameters and run metadata
        # config={
        #     "learning_rate": lr,
        #     "epochs": epochs,
        # },
    )
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")  
    ds = load_from_disk('artifacts/summary_dataset')
    def get_len(sample):
        sample['summary_len'] = len(tokenizer(sample['summary'])[0])
        sample['text_len'] = len(tokenizer(sample['text'])[0])
        return sample
    stat_ds = ds.map(get_len, num_proc=20)
    filter_ds = stat_ds.filter(
        lambda x: (
            x['summary_len'] >= 41 and
            x['summary_len'] <= 129 and
            x['text_len'] >= 370 and
            x['text_len'] <= 896
        )
    )
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
    clean_ds = filter_ds.map(clean_text)
    base_model = "VietAI/vit5-large-vietnews-summarization"
    config = AutoConfig.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)  
    model = T5ForConditionalGeneration.from_pretrained(base_model, config=config)
    model.cuda()
    def transform(batch):
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
    transform_ds = clean_ds.map(transform, num_proc=20).with_format('torch')
    ds = transform_ds.train_test_split(test_size=0.1).with_format('torch')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir='artifacts/new_summarizer',
        evaluation_strategy='steps',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        warmup_ratio=0.02,
        eval_steps=200,
        report_to='wandb',
        logging_steps=1,
        learning_rate=0.00001443,
        lr_scheduler_type='linear',
        max_steps=3000,
        generation_config=GenerationConfig(
            num_beams=1,
            top_k=6,
            top_p=0.9,
            temperature=0.6,
            max_length=129,
            min_length=41,
            num_return_sequences=1,
            repetition_penalty=1.1,
            do_sample=False,
            penalty_alpha=0.6
        )
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        data_collator=data_collator,
    )
    trainer.train()
    save_name = 'artifacts/new_summarizer_model'
    trainer.save_model(save_name)
    tokenizer.save_pretrained(save_name)
    config.save_pretrained(save_name)

def test_gradio():
    checkpoint = '/home/yuuhanase/FPTU/DAT/vi_news_chatbot/artifacts/new_summarizer_model'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    config = AutoConfig.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, config=config)
    model.cuda()
    def clean_text(batch):
        batch = batch.replace("\n", ' ')
        batch = batch.replace("\t", ' ')
        batch = batch.replace("\r", ' ')
        batch = clean_extra_whitespace(batch)

        batch = batch.replace("\n", ' ')
        batch = batch.replace("\t", ' ')
        batch = batch.replace("\r", ' ')
        batch = clean_extra_whitespace(batch)

        return batch
    def process(text):
        input = clean_text(text)
        tokenized_input = tokenizer(input, return_tensors='pt').to(model.device)
        output = model.generate(
            input_ids=tokenized_input['input_ids'],
            num_beams=1,
            top_k=6,
            top_p=0.9,
            temperature=0.6,
            max_length=129,
            min_length=41,
            num_return_sequences=1,
            repetition_penalty=1.1,
            do_sample=True,
            penalty_alpha=0.6
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        tokenized_input = tokenized_input.to('cpu')
        output = output.to('cpu')
        del tokenized_input, output
        torch.cuda.empty_cache()
        return decoded_output.replace('. ', '.\n')
    
    demo = gr.Interface(fn=process, inputs="textbox", outputs="textbox")
    demo.queue()
    demo.launch(share=True)

def insert_db():
    from dictionary.dictionary import Dictionary
    from underthesea import sent_tokenize, word_tokenize

    dictionary = Dictionary()
    dictionary.load("/home/yuuhanase/FPTU/DAT/vi_news_chatbot/experiments/dictionary/dictionary")
    word_list = [word.text for word in dictionary.words]

    def segment_str(chars, exclude=None):
        """
        Segment a string of chars using the pyenchant vocabulary.
        Keeps longest possible words that account for all characters,
        and returns list of segmented words.

        :param chars: (str) The character string to segment.
        :param exclude: (set) A set of string to exclude from consideration.
                        (These have been found previously to lead to dead ends.)
                        If an excluded word occurs later in the string, this
                        function will fail.
        """
        words = []

        if not chars.isalpha():  # don't check punctuation etc.; needs more work
            return [chars]

        if not exclude:
            exclude = set()

        working_chars = chars
        while working_chars:
            # iterate through segments of the chars starting with the longest segment possible
            for i in range(len(working_chars), 1, -1):
                segment = working_chars[:i]
                if segment.lower() in word_list and segment not in exclude:
                    words.append(segment)
                    working_chars = working_chars[i:]
                    break
            else:  # no matching segments were found
                if words:
                    exclude.add(words[-1])
                    return segment_str(chars, exclude=exclude)
                # let the user know a word was missing from the dictionary,
                # but keep the word
                print('"{chars}" not in dictionary (so just keeping as one segment)!'
                    .format(chars=chars))
                return [chars]
        # return a list of words based on the segmentation
        return words

    def seg_stuck_word(sample):
        content = sample['content']
        texts = sent_tokenize(content)
        ret = []
        for text in texts:
            words = word_tokenize(text)
            for word in words:
                for w in word.split(" "):
                    output = segment_str(w)
                    ret += output
        sample['content'] = " ".join(ret)
        return sample
    def execute(query):
        cursor.execute(query)
        conn.commit()
        return cursor
    import psycopg2
    conn = psycopg2.connect(
        database="postgres", 
        user="admin", 
        password="1234", 
        host="127.0.0.1", 
        port="6000")
    
    cursor = conn.cursor()
    execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)
    execute("""
        CREATE TABLE IF NOT EXISTS news (
            id bigserial unique,
            time integer,
            data text,
            embed vector(768),
            link text       
        );
    """)
    execute("""
        CREATE TABLE IF NOT EXISTS sentence (
            id bigserial unique,
            news_id bigserial,
            data text,
            embed vector(768)
        );
    """)
    ds = load_from_disk('/home/yuuhanase/FPTU/DAT/vi_news_chatbot/artifacts/post_content_data')

    def segment_str(chars, exclude=None, debug=False):
        """
        Segment a string of chars using the pyenchant vocabulary.
        Keeps longest possible words that account for all characters,
        and returns list of segmented words.

        :param chars: (str) The character string to segment.
        :param exclude: (set) A set of string to exclude from consideration.
                        (These have been found previously to lead to dead ends.)
                        If an excluded word occurs later in the string, this
                        function will fail.
        """
        words = []

        if not chars.isalpha():  # don't check punctuation etc.; needs more work
            return [chars]

        if not exclude:
            exclude = set()

        working_chars = chars
        while working_chars:
            # iterate through segments of the chars starting with the longest segment possible
            for i in range(len(working_chars), 1, -1):
                segment = working_chars[:i]
                if segment.lower() in word_list and segment not in exclude:
                    words.append(segment)
                    working_chars = working_chars[i:]
                    break
            else:  # no matching segments were found
                if words:
                    exclude.add(words[-1])
                    return segment_str(chars, exclude=exclude)
                # let the user know a word was missing from the dictionary,
                # but keep the word
                if debug:
                    print('"{chars}" not in dictionary (so just keeping as one segment)!'
                        .format(chars=chars))
                return [chars]
        # return a list of words based on the segmentation
        return words

    def seg_stuck_word(sample):
        content = sample['content']
        texts = sent_tokenize(content)
        ret = []
        for text in texts:
            words = word_tokenize(text)
            for word in words:
                for w in word.split(" "):
                    output = segment_str(w)
                    ret += output
        sample['content'] = " ".join(ret)
        return sample
    normalize_ds = ds.map(seg_stuck_word, num_proc=20)
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    import py_vncorenlp
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/yuuhanase/FPTU/DAT/vi_news_chatbot/experiments/VnCoreNLP')
    
    def insert_db(batch, idx):
        title = batch['title'].replace(" - VnExpress", "")
        title_input = ' '.join(rdrsegmenter.word_segment(title))
        time = batch['unix_timestamp']
        title_e = [float(e) for e in model.encode(title_input)]
        link = batch['article_links']
        execute(f"""
            INSERT INTO news(id, time, data, embed, link)
            VALUES (
                {idx},
                {time},
                '{title.replace("'", "''")}',
                '{title_e}',
                '{link}'
            )
        """)
        del title, time, title_e
        content = batch['content']
        for sent in sent_tokenize(content):
            sent_input = ' '.join(rdrsegmenter.word_segment(sent))
            sent_e = [float(e) for e in model.encode(sent_input)]
            execute(f"""
                INSERT INTO sentence(news_id, data, embed)
                VALUES (
                    {idx},
                    '{sent.replace("'", "''")}',
                    '{sent_e}'
                )
            """)

    normalize_ds.map(insert_db, with_indices=True)
    conn.close()

def pipeline():
    def search_post_link(query):
        def clean_input(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.replace(" ", "+")
            print(text)
            return text
        url = f'https://timkiem.vnexpress.net/?q={clean_input(query)}'
        delay = 0
        while True:
            response = requests.get(url)
            if response.ok:
                break
            time.sleep(0.5 + delay)
            delay += 0.1
            if 0.5 + delay > 1.0:
                print("Time out")
                raise Exception("Time out")
        soup = BeautifulSoup(response.content, "html.parser")
        titles = []
        links = []
        for article in soup.find_all("div", attrs="width_common list-news-subfolder")[0].find_all("article"):
            if article.h3:
                titles.append(article.h3.a.get('title'))
                links.append(article.h3.a.get("href"))
        return titles, links
    def choose_most_related_news(titles):
        from sklearn.cluster import KMeans
        import numpy as np
        from sklearn.metrics import silhouette_score
        sentences = [" ".join(rdrsegmenter.word_segment(sent)) for sent in titles]
        embeddings = search_model.encode(sentences)
        silhouette_avg = []
        for num_clusters in list(range(2,len(titles))):
            kmeans = KMeans(n_clusters=num_clusters, init = "k-means++", n_init = 10)
            kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            silhouette_avg.append(score)
        best_k = np.argmax(silhouette_avg)+2
        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init="auto").fit(embeddings)
        y_kmeans = kmeans.predict(embeddings)
        _, counts = np.unique(y_kmeans, return_counts=True)
        return np.where(y_kmeans == np.argmax(counts))
    def crawl_content(url):    
        delay = 0.0
        ret = {}
        try:
            while True:
                response = requests.get(url)
                if response.ok:
                    break
                time.sleep(0.5 + delay)
                delay += 0.1
                if 0.5 + delay > 1.0:
                    print("Time out")
                    raise Exception("Time out")
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.text
            article_0 = []
            article_1 = []
            for p in soup.find_all("body")[0].find_all("p"):
                if p.text not in article_0 and not p.find("strong"):
                    article_0.append(p.text)
            for p in soup.find_all("p", "Normal"):
                if p.text not in article_1 and not p.find("strong"):
                    article_1.append(p.text)
            article = article_0 if len(article_0) >= len(article_1) else article_1
            content = " ".join(article)
            ret['title'] = title
            ret['content'] = content
            ret['link'] = url
        except Exception as e:
            print(e)
            ret['title'] = ""
            ret['content'] = ""
            ret['link'] = url
        return ret 

    def execute(query):
        cursor.execute(query)
        conn.commit()
        return cursor
    from sentence_transformers import SentenceTransformer
    import py_vncorenlp
    import psycopg2
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/yuuhanase/FPTU/DAT/vi_news_chatbot/experiments/VnCoreNLP')
    search_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    checkpoint = '/home/yuuhanase/FPTU/DAT/vi_news_chatbot/artifacts/new_summarizer_model'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    config = AutoConfig.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, config=config)
    model.cuda()
    # conn = psycopg2.connect(
    #     database="postgres", 
    #     user="admin", 
    #     password="1234", 
    #     host="127.0.0.1", 
    #     port="6000")
    # cursor = conn.cursor()
    def clean_text(batch):
        batch = batch.replace("\n", ' ')
        batch = batch.replace("\t", ' ')
        batch = batch.replace("\r", ' ')
        batch = clean_extra_whitespace(batch)
        return batch

    def summarize(text):
        input = clean_text(text)
        tokenized_input = tokenizer(input, return_tensors='pt').to(model.device)
        output = model.generate(
            input_ids=tokenized_input['input_ids'],
            num_beams=1,
            top_k=6,
            top_p=0.9,
            temperature=0.6,
            max_length=129,
            min_length=41,
            num_return_sequences=1,
            repetition_penalty=1.1,
            do_sample=True,
            penalty_alpha=0.6
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        tokenized_input = tokenized_input.to('cpu')
        output = output.to('cpu')
        del tokenized_input, output
        torch.cuda.empty_cache()
        return decoded_output.replace('. ', '.\n')
    def process(text):
        # text = " ".join(rdrsegmenter.word_segment(text))
        # text_e = [float(e) for e in search_model.encode(text)]
        # ret = execute(f"""
        #     SELECT id, data, link
        #     FROM news
        #     WHERE id IN (
        #         WITH news_search AS (
        #             SELECT id, data, embed <=> '{text_e}' 
        #             FROM news
        #             ORDER BY embed <=> '{text_e}' 
        #         ), sentence_search AS (
        #             SELECT news_id, data
        #             FROM sentence
        #             ORDER BY embed <=> '{text_e}' 
        #         )
        #         SELECT ns.id
        #         FROM news_search ns, sentence_search ss
        #         WHERE ns.id = ss.news_id
        #         LIMIT 20
        #     )
        #     LIMIT 10;
        # """)
        # news_data = []
        # for result in ret.fetchall():
        #     news_id, title, link = result
        #     content = execute(f"""
        #         SELECT data
        #         FROM sentence
        #         WHERE news_id = {news_id};
        #     """).fetchall()
        #     content = " ".join([sent[0] for sent in content])
        #     summary = summarize(content)
        #     news_data.append({
        #         'title': title,
        #         'content': content,
        #         'summary': summary,
        #         'link': link
        #     })
        titles, links = search_post_link(text)
        idx = choose_most_related_news(titles)
        links = np.array(links)
        related_links = links[idx]
        news_data = []
        for i, link in enumerate(related_links):
            print(i)
            ret = crawl_content(link)
            ret['summary'] = summarize(ret['content'])
            news_data.append(ret)
        big_news = "\n".join([data['summary'] for data in news_data])
        # big_news_summary = summarize(big_news)
        ref_links = '\n\t'.join([data['link'] for data in news_data])
        ret_format = f"""{big_news}

Nguồn:
\t{ref_links}
"""
        return ret_format
    demo = gr.Interface(fn=process, inputs="textbox", outputs="textbox")
    demo.queue()
    demo.launch(share=True)
        
if __name__ == '__main__':
    # crawl_post_link()
    # crawl_post_content()
    # create_dataset()
    # train()
    # test_gradio()
    # insert_db()
    pipeline()