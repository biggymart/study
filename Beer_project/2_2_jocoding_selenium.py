# Crawling: 인터넷에 있는 정보를 자동으로 다운로드
# aka parsing, spidering

# BeautifulSoup (HTML parser), Requests, 구름IDE(ide.goorm.io)
# $python index.py

# pip install bs4
from bs4 import BeautifulSoup
from urllib.request import urlopen

with urlopen('http:/en.wikipedia.org/wiki/Main_Page') as response:
    soup = BeautifulSoup(response, 'html.parser')
    for anchor in soup.find_all('a'):
        print(anchor.get('href','/'))

# f12 개발자 화면, Beautifulsoup 공식문서
# .select
# soup.select("span.ah_k") <span class="ah_k">를 선택하라는 뜻
# .get_text()

f = open("C:/doit/새파일.txt", 'w')
for i in range(1, 11):
    f.write(i)
f.close()

# 이미지 주소: src

# pip install google_images_download