#!/usr/bin/env python 
#coding:utf-8 
  
try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 
import sys 


class Entry:
	def __init__(self):
		self.SEN = ''
		self.NAM = []
		self.PRO = []

def readXml(xmlfile):
	tree = ET.ElementTree(file = xmlfile)
	entities = []
	corefs = []
	for entry in tree.getroot():
		extent = entry.findall('entity/entity_mention')
		for ext in extent:
			extType = ext.get('TYPE')
			entity = ext.find('extent/charseq').text
			# print entity
			if entity == '':
				print "error context:",xmlfile
				sys.exit(1)
			else:
				entities.append(entity)
				if extType == 'PRO':
					corefs.append(entity)
	if len(entities) != 3:
		print "error in dataset"
	nam = [entities[0:2]] + [entities[2:3]]
	return nam,corefs


def readsgm(sgmfile):
	lines = open(sgmfile,'r').readlines()
	for i in range(0,len(lines)):
		if lines[i] == '<TEXT>\n':
			text = lines[i+1][0:-1]
			return text
	return ""
   

def readDataFromFiles(pathDir):
	trainPaths = pathDir + 'train.txt'
	testPaths = pathDir + 'test.txt'
	dataPaths = pathDir + 'data/'

	trainFiles = open(trainPaths,'r').readlines()
	testFiles = open(testPaths,'r').readlines()

	filelists = open(dataPaths+'fileList.txt','r').readlines()

	entries = []
	trainIds = []
	testIds = []

	for fl in filelists:
		if fl != '\n':
			apf = dataPaths + fl[0:-1]
			sgm = dataPaths + fl.split('.')[0] + '.sgm'
			sentence = readsgm(sgm)
			ent = Entry()
			ent.SEN = sentence
			(ent.NAM, ent.PRO) = readXml(apf)
			# print ent.NAM, ent.PRO
			entries.append(ent)

	for tFile in trainFiles:
		if tFile != '\n':
			trainIds.append(tFile.split('.')[0])

	for tFile in testFiles:
		if tFile != '\n':
			testIds.append(tFile.split('.')[0])

	print len(entries),len(trainIds),len(testIds)

	return entries,trainIds,testIds

if __name__ == '__main__':
	readDataFromFiles("/home/ludc/Winograd/data/WinoCoref/")