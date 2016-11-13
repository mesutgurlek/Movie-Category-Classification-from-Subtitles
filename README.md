-The project folder in scrapy_project includes a web-crawler built with Scrapy Framework

- In order to be able to execute the project

pip install scrapy

More info: https://doc.scrapy.org/en/latest/intro/install.html#intro-install

- Then execute the following command in the scrapy_project folder:

scrapy crawl subtitles

- You need to edit the 'subtitles_path' variable at the top of the subtitles_spider.py
You should configure the path so that script downloads the subtitles into that folder

- Beware the download limit which is 200 downloads/per day !!

