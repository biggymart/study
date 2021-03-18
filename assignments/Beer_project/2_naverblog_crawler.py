# https://bskyvision.com/721
# pip install beautifulsoup4
# https://ultrakid.tistory.com/13

import os
cwd = os.getcwd()
print(cwd)

from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus
 
baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
plusUrl = input('검색어 입력: ')
crawl_num = int(input('크롤링할 갯수 입력(최대 50개): '))
 
url = baseUrl + quote_plus(plusUrl) # 한글 검색 자동 변환
html = urlopen(url)
soup = bs(html, "html.parser")
img = soup.find_all(class_='_img')
 
n = 1
for i in img:
    print(n)
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f:
        with open('./data/image/letter/img' + str(n)+'.jpg','wb') as h: # w - write b - binary
            img = f.read()
            h.write(img)
    n += 1
    if n > crawl_num:
        break
    
    
print('Image Crawling is done.')

# 인공지능신문
# https://www.aitimes.kr/news/articleView.html?idxno=15924

# 글자체 이미지 데이터
# https://www.eiric.or.kr/special/special.php
# https://blog.naver.com/PostView.nhn?blogId=sogangori&logNo=221432815643