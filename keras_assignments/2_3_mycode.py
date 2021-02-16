# 라이브러리 다 불러와 (셀레니움 고고)
# 후보: 하이트 / 카스 / 테라 / 클라우드 / 맥스 / 필라이트 / 필굿

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

import os
import time

WORD_LST = [
    '카스 캔맥주',
    '하이트 캔맥주',
    '테라 캔맥주',
    '클라우드 캔맥주',
    '맥스 캔맥주',
    '필라이트 캔맥주',
    '필굿 캔맥주'
]

def get_url(WORD):
    URL = 'http://www.google.com/search?q=' + WORD + '&source=lnms&tbm=isch'
    return URL

browser = webdriver.Chrome('C:/Users/ai/Downloads/chromedriver_win32/chromedriver.exe')

# 이쁘게 이쁘게~
browser.set_sindow_size(1280, 1024)
browser.get(get_url(get_url()))
time.sleep(1)



# 디렉토리 없으면 만들어주는 거 (optioinal, 시간 남으면 만들기)
# dirs = 'images'
# if not os.path.exists(dirs):
#     os.mkdir(dirs)


# 데이터 수집할 도구 준비 (셀레니움 등)


# 데이터 전처리


# 모델링


# 출력 및 예측