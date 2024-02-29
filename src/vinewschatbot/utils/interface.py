from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import string
import time
import requests
from bs4 import BeautifulSoup
from unstructured.cleaners.core import clean_extra_whitespace
import torch

def clean_input(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace(" ", "+")
    print(text)
    return text

def search_post_link(query):
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

def choose_most_related_news(titles, rdrsegmenter, search_model):
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

def clean_text(batch):
    batch = batch.replace("\n", ' ')
    batch = batch.replace("\t", ' ')
    batch = batch.replace("\r", ' ')
    batch = clean_extra_whitespace(batch)
    return batch

def summarize(text, tokenizer, model):
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