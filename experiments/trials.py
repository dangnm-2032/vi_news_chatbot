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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  
import torch
import gradio as gr
from underthesea import sent_tokenize
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
    while True:
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
                break
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
    data_link_path = "/home/yuuhanase/workspace/DangProject/vi_news_summarizer/artifacts/yesterday_data"
    crawl_data = load_from_disk(data_link_path)
    content_data = crawl_data.map(transform, batched=False)
    content_data.save_to_disk("/home/yuuhanase/workspace/DangProject/vi_news_summarizer/artifacts/yesterday_content_data")
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

if __name__ == '__main__':
    # crawl_post_link()
    create_dataset()