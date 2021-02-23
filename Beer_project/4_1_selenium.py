from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

import json
import os
import sys
import argparse

import requests
import urllib
import urllib3
import time

###1. crawling from Naver and Google
searchword1 = '하이트'  # 카스, 하이트, 필라이트, 필굿
searchword2 = '캔맥주'
dirs = 'C:/data/image/beer_selenium/hite/'
searchurl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + searchword1 + '+' + searchword2
# searchurl = 'https://www.google.com/search?q=' + searchword1 + '+' + searchword2 + '&source=lnms&tbm=isch'
maxcount = 1000

browser = webdriver.Chrome('path_to_driver/chromedriver.exe')
browser.get(searchurl)
time.sleep(1)

element = browser.find_element_by_tag_name('body')
# Scroll down 1
for i in range(30):
    element.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.3)
try:
    browser.find_element_by_id('smb').click() # 결과 더보기
    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)
except:
    for i in range(10):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

try:
    browser.find_element_by_xpath('//input[@value="결과 더보기"]').click()
except:
    pass

# Scroll down 2
# 위에 있는 코드와 동일

page_source = browser.page_source 
soup = BeautifulSoup(page_source, 'lxml')
images = soup.find_all('img')

urls = []
for image in images:
    try:
        url = image['data-src']
        if not url.find('https://'):
            urls.append(url)
    except:
        try:
            url = image['src']
            if not url.find('https://'):
                urls.append(image['src'])
        except Exception as e:
            print(f'No found image sources.')
            print(e)

count = 0
if urls:
    for url in urls:
        try:
            if count > maxcount:
                break
            res = requests.get(url, verify=False, stream=True)
            rawdata = res.raw.read()
            with open(os.path.join(dirs, 'img_' + str(count) + '.jpg'), 'wb') as f:
                f.write(rawdata)
                count += 1
        except Exception as e:
            print('Failed to write rawdata.')
            print(e)

browser.close()
print(count, "장 다운로드")