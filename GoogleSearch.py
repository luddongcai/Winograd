import requests
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(description='Get Google Count.')
parser.add_argument('word', help='word to count')
args = parser.parse_args()

print (args.word)

r = requests.get('http://www.google.com/search',
                 params={'q':'"'+args.word+'"',
                         "tbs":"li:1"}
                )

soup = BeautifulSoup(r.text,"lxml")

print (soup.find('div',{'id':'resultStats'}).text)
