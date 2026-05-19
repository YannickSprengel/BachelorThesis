from fundus import Crawler, PublisherCollection, Article
from transformers import AutoTokenizer
crawler = Crawler(PublisherCollection)

for article in crawler.crawl(max_articles=1):
    print(article)
    html = article.html
    text = article.plaintext

    print(html)