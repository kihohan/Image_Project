# 김치 / 만두 / 감자탕 / 족발 / 김밥 / 죽
from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from selenium import webdriver

keyword = str(input('keyword: '))
path = str(input('path: '))
# "/home/kiho/다운로드/두잉랩_과제/"
url="https://www.google.com/search?q="+keyword+"&sxsrf=ALeKk00jiU8Mzgqk26orOnCqs466KTK1ow:1594882982524&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjhsIaNmtHqAhWqUN4KHTE7AwkQ_AUoAXoECBYQAw&biw=1853&bih=977"
driver = webdriver.Chrome("/home/kiho/다운로드/chromedriver")
driver.get(url)
for i in range(1000):
    driver.execute_script("window.scrollBy(0,10000)")
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
img = soup.select('.rg_i.Q4LuWd')
n = 1
imgurl = []

for i in img:
    try:
        imgurl.append(i.attrs["src"])
    except KeyError:
        imgurl.append(i.attrs["data-src"])

for i in imgurl:
    urlretrieve(i, path + keyword + str(n) + ".jpg")
    n+=1

    if(n == 400): 
        break

driver.close()
