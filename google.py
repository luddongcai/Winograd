import requests
from bs4 import BeautifulSoup
import argparse
import sys

arg_temp=''
for i in range(1, len(sys.argv)):
	arg_temp=arg_temp+sys.argv[i]
	if i !=len(sys.argv)-1:
		arg_temp+='+'
#parser = argparse.ArgumentParser(description='Get Google Count.')
#parser.add_argument('word', help='word to count')
#args = parser.parse_args()

print('url:'+arg_temp)
# r = requests.get('https://www.google.com/search?q=bent+bee&lr=lang_zh-CN&tbs=li:1')
#r = requests.get('https://www.google.com.hk/search?q=bee&lr=lang_zh-CN&tbs=li:1&gws_rd=cr&ei=cgmgV52GEoyBvgTOlZKgAg')
r = requests.get('http://www.google.com/search',
                 params={'q':arg_temp,
                        "tbs":"li:1"}
                )
soup = BeautifulSoup(r.text)
print(soup.find('div',{'id':'resultStats'}).text)
