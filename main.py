#!/usr/bin/env python 
#coding:utf-8 

# Wrote by Dongcai Lu

# sys.exit(1)
# print(output)
# output = nlp.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
# print(output)
# output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
# print(output)

from pycorenlp import StanfordCoreNLP
from nltk.corpus import wordnet as wn
import readFile
import sys
import json
import readline
import Schema
import re

path = "/home/ludc/Winograd/data/WinoCoref/"
outputfile = "/home/ludc/Winograd/python/log/"
knowledgefile = "/home/ludc/Winograd/python/knowledge/"
Schema1File = "/home/ludc/Winograd/python/knowledge/KnowledgeBaseSchema1.txt"
Schema2File = "/home/ludc/Winograd/python/knowledge/KnowledgeBaseSchema2.txt"

def replaceReference(text,coref):
    processText = text.replace('?', ' ?').replace('.',' .').replace(',', ' ,').replace('!',' !') + ' '
    pronoun = ' ' + coref[0] + ' '
    noun = ' ' + coref[1] + ' '
    return processText.replace(pronoun, noun)

def FindCausalWords(text):
    processText = ' ' + text.lower().replace('?', ' ?').replace('.',' .').replace(',', ' ,').replace('!',' !') + ' '
    for word in Schema.CausalWords:
        token = ' ' + word + ' '
        if processText.find(token) != -1:
            return word
    return ""


def Simility_Word(word1,word2):
    word1Syn = wn.synsets(word1)
    word2Syn = wn.synsets(word2)
    WholeSimi = 0
    for syn1 in word1Syn:
        maxValue = 0
        for syn2 in word2Syn:
            simi = syn1.lin_similarity(syn2,brown_ic)
            if maxValue < simi:
                maxValue = simi
        WholeSimi += maxValue
    for syn2 in word2Syn:
        maxValue = 0
        for syn1 in word1Syn:
            simi = syn2.lin_similarity(syn1,brown_ic)
            if maxValue < simi:
                maxValue = simi
        WholeSimi += maxValue
    if (len(word1Syn)+len(word2Syn)) == 0:
        return 0
    else:
        return WholeSimi/(len(word1Syn)+len(word2Syn))

def Simility_XJK_Word(word1,word2):
    word1Syn = wn.synsets(word1)
    word2Syn = wn.synsets(word2)
    maxValue = 0
    for syn1 in word1Syn:
        for syn2 in word2Syn:
            if syn1.pos() == syn2.pos():
                simi = syn1.path_similarity(syn2)
                if maxValue < simi:
                    maxValue = simi
    return maxValue

def SimilityBetweenPhrases(phrase1,phrase2):
    words1 = phrase1.split()
    words2 = phrase2.split()
    simi = 0
    for w1 in words1:
        for w2 in words2:
            simi += Simility_XJK_Word(w1,w2)

    # print "simility between ", phrase1, ' and ', phrase2, ': ', simi/(len(words1)*len(words2))
    return simi/(len(words1) * len(words2))

def CalculateKnowledgeValue(relations, type2KnowledgeIndex, causal):
    if (causal not in type2KnowledgeIndex) or len(relations) == 0:
        return 0
    else:
        sumValue = 0
        # print len(relations)
        for relation in relations:
            prev = relation.split('|')[0]
            back = relation.split('|')[1]
            m = re.search(r'(.+)\((.+),(.+)\)',prev)
            n = re.search(r'(.+)\((.+),(.+)\)',back)
            print m.group(1),m.group(2),m.group(3)
            print n.group(1),n.group(2),n.group(3)
            maxValue = 0
            maxKnowledge = ''
            for type2 in type2KnowledgeIndex[causal]:
                tprev = type2.split('|')[0]
                tback = type2.split('|')[1]
                tm = re.search(r'(.+)\((.+),(.+)\)',tprev)
                tn = re.search(r'(.+)\((.+),(.+)\)',tback)
                if m.group(2) == tm.group(2) and m.group(3) == tm.group(3) and \
                    n.group(2) == tn.group(2) and n.group(3) == tn.group(3):

                    # print tm.group(1),tn.group(1)

                    value = SimilityBetweenPhrases(m.group(1),tm.group(1)) * SimilityBetweenPhrases(n.group(1),tn.group(1))

                    # print value
                    if value > maxValue:
                        maxKnowledge = type2
                        maxValue = value
            print maxKnowledge
            sumValue += maxValue

        return sumValue/(len(relations))



def ConstructKnowledgeFromTrainData(datasets,TrainSetsId,knowfile):
    knowledgeBaseWriter = open(knowledgefile + knowfile, 'w')
    Schema1FileWriter = open(Schema1File, 'w')
    Schema2FileWriter = open(Schema2File, 'w')
    for trainId in TrainSetsId:
        print (trainId)
        text = datasets[int(trainId)].SEN
        text = replaceReference(text,datasets[int(trainId)].NAM[0])
        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,openie,dcoref',
            'outputFormat': 'json'
        })

        ### Getting the coreference results.
        corefs = []
        for key in output["corefs"].keys():
            # print output["corefs"][key]
            # print len(output["corefs"][key])
            corpus = []
            for enti in output["corefs"][key]:
                # print enti['text']
                corpus.append(str(enti['text']))
            corefs.append(corpus)
        # print corefs
        fileResult = open(outputfile + str(trainId), 'w')
        fileResult.write('Sentence:\t' + text + '\n')
        fileResult.write('Stanford:\t'+str(corefs) + '\n')
        fileResult.write('Winograd:\t'+str(datasets[int(trainId)].NAM))
        fileResult.close()

        ### extracting knowledge base using stanford openie
        # print text
        causal = FindCausalWords(text)
        tmpRelations = []
        knowledgeBaseWriter.write("sentence: " + text + '\n')
        knowledgeBaseWriter.write("Relation: ")
        for sent in output["sentences"]:
            # for key in sent.keys():
            #     print key
            tokens = {}
            lemmas = {}
            for token in sent['tokens']:
                # print token['word'], '---', token['lemma'], '---', token['pos']
                tokens[str(token['word'])] = str(token['lemma'])
                lemmas[str(token['lemma'])] = str(token['word'])
            # print tokens
            for openie in sent['openie']:
                OriKnowledge = openie['relation'] + '(' + openie['subject'] + ',' + openie['object'] + ')'
                # print OriKnowledge
                _relation = str(openie['relation'])
                _subject = str(openie['subject'])
                _object = str(openie['object'])
                splR = _relation.split()
                splS = _subject.split()
                splO = _object.split()

                for tmp in splR:
                    if tmp in tokens:
                        _relation = _relation.replace(tmp, tokens[tmp])
                for tmp in splS:
                    if tmp in tokens:
                        _subject = _subject.replace(tmp, tokens[tmp])
                for tmp in splO:
                    if tmp in tokens:
                        _object = _object.replace(tmp, tokens[tmp])
                LemKnowledge = _relation + '(' + _subject + ',' + _object + ')'
                Schema1FileWriter.write(LemKnowledge + '\n')
                tmptmp = tmpRelations
                AppendFlag = True
                for tmpRel in tmptmp:
                    rel = re.search(r'(.+)\((.+),(.+)\)',tmpRel)
                    if rel.group(1).find(_relation) != -1 and rel.group(2).find(_subject) != -1 and rel.group(3).find(_object) != -1:
                        tmpRelations.remove(tmpRel)
                        # tmpRelations.append(LemKnowledge)
                        break
                    elif _relation.find(rel.group(1)) != -1 and _subject.find(rel.group(2)) != -1 and _object.find(rel.group(3)) != -1:
                        AppendFlag = False
                        break
                    else:
                        continue
                if AppendFlag == True:
                    tmpRelations.append(LemKnowledge)
                knowledgeBaseWriter.write(LemKnowledge + '.')
        if causal != '':
            for i in range(0,len(tmpRelations)):
                for j in range(i+1,len(tmpRelations)):
                    m = re.search(r'(.+)\((.+),(.+)\)',tmpRelations[i])
                    n = re.search(r'(.+)\((.+),(.+)\)',tmpRelations[j])
                    # for tmp in m.group(1).split():
                    #     if tmp in lemmas:
                    #         mRel = m.group(1).replace(tmp, lemmas[tmp])
                    # for tmp in n.group(1).split():
                    #     if tmp in lemmas:
                    #         nRel = n.group(1).replace(tmp, lemmas[tmp])
                    if m.group(2) == n.group(2):
                        tempelate = "(" + m.group(1) + '(' + 'X' + ',' + m.group(3) +')|'
                        tempelate += n.group(1) + '(' + 'X' + ',' + n.group(3) +'). X=' + m.group(2) +'. '+ causal + ")\n"
                        Schema2FileWriter.write(tempelate)
                    elif m.group(2) == n.group(3):
                        tempelate = "(" + m.group(1) + '(' + 'X' + ',' + m.group(3) +')|'
                        tempelate += n.group(1) + '(' + n.group(2) + ',' + 'X' +'). X=' + m.group(2) +'. '+ causal + ")\n"
                        Schema2FileWriter.write(tempelate)
                    elif m.group(3) == n.group(2):
                        tempelate = "(" + m.group(1) + '(' + m.group(2) + ',' + 'X' +')|'
                        tempelate += n.group(1) + '(' + 'X' + ',' + n.group(3) +'). X=' + m.group(3) +'. '+ causal + ")\n"
                        Schema2FileWriter.write(tempelate)
                    elif m.group(2) == n.group(2):
                        tempelate = "(" + m.group(1) + '(' + m.group(2) + ',' + 'X' +')|'
                        tempelate += n.group(1) + '(' + n.group(2) + ',' + 'X' +'). X=' + m.group(3) +'. '+ causal + ")\n"
                        Schema2FileWriter.write(tempelate)
        knowledgeBaseWriter.write(causal + '\n')


    knowledgeBaseWriter.close()
    Schema1FileWriter.close()
    Schema2FileWriter.close()

def TestCorefs(text):
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,ner,lemma,depparse,natlog,openie,dcoref',
        'outputFormat': 'json'
    })
    corefs = []
    for key in output["corefs"].keys():
        # print output["corefs"][key]
        # print len(output["corefs"][key])
        corpus = []
        for enti in output["corefs"][key]:
            # print enti['text']
            corpus.append(str(enti['text']))
        corefs.append(corpus)
    print ("entities list:",corefs)

    # tmpRelations = []

    ### extracting knowledge base using stanford openie
    for sent in output["sentences"]:
        # for key in sent.keys():
        #     print key
        tokens = {}
        for token in sent['tokens']:
            print (token['ner'])
            # print token['word'], '---', token['lemma'], '---', token['pos']
            tokens[token['word']] = token['lemma']
        for openie in sent['openie']:
            OriKnowledge = openie['relation'] + '(' + openie['subject'] + ',' + openie['object'] + ')'
            # print OriKnowledge
            _relation = str(openie['relation'])
            _subject = str(openie['subject'])
            _object = str(openie['object'])
            splR = _relation.split()
            splS = _subject.split()
            splO = _object.split()
            for tmp in splR:
                if tmp in tokens:
                    _relation = _relation.replace(tmp, tokens[tmp])
            for tmp in splS:
                if tmp in tokens:
                    _subject = _subject.replace(tmp, tokens[tmp])
            for tmp in splO:
                if tmp in tokens:
                    _object = _object.replace(tmp, tokens[tmp])
            LemKnowledge = _relation + '(' + _subject + ',' + _object + ')'
            print ("knowledge extracted from openie: ",LemKnowledge)


def isSame(string1, string2):
    lowstring1 = string1.lower()
    lowstring2 = string2.lower()
    stopwords = ['the','a','an','and','or','of','in']
    words1 = lowstring1.split()
    words2 = lowstring2.split()

    for word1 in words1:
        if word1 not in stopwords and word1 in words2:
            # print string1,string2
            return True
    return False

def SolvingCoreference(DataSets,TestSetsId,type1File,type2File):
    # TestSentWriter = open(knowledgefile + "TestSent.txt", 'w')
    type1Knowledge = open(type1File, 'r').readlines()
    type2Knowledge = open(type2File, 'r').readlines()
    print (len(type1Knowledge), len(type2Knowledge))

    type2KnowledgeIndex = {}
    
    for t2k in type2Knowledge: ###Normalization
        tuples = t2k[1:-2].split('. ')
        twoRelations = tuples[0].split('|')
        m = re.search(r'(.+)\((.+),(.+)\)',twoRelations[0])
        if m.group(2) != 'X':
            NormalizePrevRel = twoRelations[0].replace(m.group(2), 'Y')
        elif  m.group(3) != 'X':
            NormalizePrevRel = twoRelations[0].replace(m.group(3), 'Y')
        NormalizePrevRel += '|'
        m = re.search(r'(.+)\((.+),(.+)\)',twoRelations[1])
        if m.group(2) != 'X':
            NormalizePrevRel += twoRelations[1].replace(m.group(2), 'Z')
        elif  m.group(3) != 'X':
            NormalizePrevRel += twoRelations[1].replace(m.group(3), 'Z')

        # print NormalizePrevRel, tuples[2]
        if tuples[2] not in type2KnowledgeIndex:
            type2KnowledgeIndex[tuples[2]] = []
            type2KnowledgeIndex[tuples[2]].append(NormalizePrevRel)
        else:
            type2KnowledgeIndex[tuples[2]].append(NormalizePrevRel)

    LogForNoRelations = []
    LogForNoRelationsWithPronoun = []

    for testId in TestSetsId:
        print (testId)
        text = DataSets[int(testId)].SEN
        OrigText = text
        # TestSentWriter.write(text + '\n')
        # text = replaceReference(text,datasets[int(trainId)].NAM[0])
        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,openie,dcoref',
            'outputFormat': 'json'
        })

        tmpRelations = []
        causal = FindCausalWords(text)

        for sent in output["sentences"]:
            tokens = {}
            lemmas = {}
            for token in sent['tokens']:
                # print token['ner']
                # print token['word'], '---', token['lemma'], '---', token['pos']
                tokens[str(token['word'])] = str(token['lemma'])
                lemmas[str(token['lemma'])] = str(token['word'])
                OrigText = OrigText.replace(str(token['word']),str(token['lemma']))

            print text,'\n',OrigText

            for openie in sent['openie']:
                OriKnowledge = openie['relation'] + '(' + openie['subject'] + ',' + openie['object'] + ')'
                # print OriKnowledge
                _relation = str(openie['relation'])
                _subject = str(openie['subject'])
                _object = str(openie['object'])
                splR = _relation.split()
                splS = _subject.split()
                splO = _object.split()
                for tmp in splR:
                    if tmp in tokens:
                        _relation = _relation.replace(tmp, tokens[tmp])
                for tmp in splS:
                    if tmp in tokens:
                        _subject = _subject.replace(tmp, tokens[tmp])
                for tmp in splO:
                    if tmp in tokens:
                        _object = _object.replace(tmp, tokens[tmp])
                LemKnowledge = _relation + '(' + _subject + ',' + _object + ')'
                # print "knowledge extracted from openie: ",LemKnowledge

                tmptmp = tmpRelations
                AppendFlag = True
                for tmpRel in tmptmp:
                    rel = re.search(r'(.+)\((.+),(.+)\)',tmpRel)
                    if rel.group(1).find(_relation) != -1 and rel.group(2).find(_subject) != -1 and rel.group(3).find(_object) != -1:
                        tmpRelations.remove(tmpRel)
                        # tmpRelations.append(LemKnowledge)
                        break
                    elif _relation.find(rel.group(1)) != -1 and _subject.find(rel.group(2)) != -1 and _object.find(rel.group(3)) != -1:
                        AppendFlag = False
                        break
                    else:
                        continue
                if AppendFlag == True:
                    tmpRelations.append(LemKnowledge)
            print str(tmpRelations),causal

        pronoun = DataSets[int(testId)].NAM[0][0]
        candidateNouns = [DataSets[int(testId)].NAM[0][1], DataSets[int(testId)].NAM[1][0]]

        # print pronoun,candidateNouns

        if len(tmpRelations) == 0:
            LogForNoRelations.append(testId)
            continue
        else:
            HavePronounRelation = False
            for tmpRel in tmpRelations:
                m = re.search(r'(.+)\((.+),(.+)\)',tmpRel)
                if m.group(2) != pronoun and m.group(3) != pronoun:
                    continue
                else:
                    HavePronounRelation = True
                    break
            if not HavePronounRelation:
                LogForNoRelationsWithPronoun.append(testId)
                continue
        if causal != '':
            candidateRelations = {}
            candidateRelations[candidateNouns[0]] = []
            candidateRelations[candidateNouns[1]] = []
            # print candidateNouns[0],candidateNouns[1]
            for i in range(0,len(tmpRelations)):
                # print tmpRelations[i]
                for j in range(i+1,len(tmpRelations)):
                    # print tmpRelations[j]
                    m = re.search(r'(.+)\((.+),(.+)\)',tmpRelations[i])
                    n = re.search(r'(.+)\((.+),(.+)\)',tmpRelations[j])
                    if (m.group(2) == pronoun and isSame(n.group(2),candidateNouns[0])) or \
                       (isSame(m.group(2),candidateNouns[0]) and n.group(2) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(X,Y)|' + m.group(1) + '(X,Z)'
                        else:
                            tempelate = m.group(1) + '(X,Y)|' + n.group(1) + '(X,Z)'
                        candidateRelations[candidateNouns[0]].append(tempelate)
                    if (m.group(2) == pronoun and isSame(n.group(2), candidateNouns[1])) or \
                         (n.group(2) == pronoun and isSame(m.group(2), candidateNouns[1])):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(X,Y)|' + m.group(1) + '(X,Z)'
                        else:
                            tempelate = m.group(1) + '(X,Y)|' + n.group(1) + '(X,Z)'
                        candidateRelations[candidateNouns[1]].append(tempelate)
                    if (m.group(3) == pronoun and isSame(n.group(2), candidateNouns[0])) or \
                        (isSame(m.group(3), candidateNouns[0]) and m.group(2) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(X,Y)|' + m.group(1) + '(Z,X)'
                        else:
                            tempelate = m.group(1) + '(Y,X)|' + n.group(1) + '(X,Z)'
                        # tempelate = m.group(1) + '(Y,X)|' + n.group(1) + '(X,Z)'
                        candidateRelations[candidateNouns[0]].append(tempelate)
                    if (m.group(3) == pronoun and isSame(n.group(2), candidateNouns[1])) or \
                        (isSame(m.group(3) , candidateNouns[1]) and n.group(2) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(X,Y)|' + m.group(1) + '(Z,X)'
                        else:
                            tempelate = m.group(1) + '(Y,X)|' + n.group(1) + '(X,Z)'
                        candidateRelations[candidateNouns[1]].append(tempelate)
                    if (m.group(2) == pronoun and isSame(n.group(3) , candidateNouns[0])) or \
                        (isSame(m.group(2) , candidateNouns[0]) and n.group(3) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(Y,X)|' + m.group(1) + '(X,Z)'
                        else:
                            tempelate = m.group(1) + '(X,Y)|' + n.group(1) + '(Z,X)'
                        candidateRelations[candidateNouns[0]].append(tempelate)
                    if (m.group(2) == pronoun and isSame(n.group(3) , candidateNouns[1])) or \
                        (isSame(m.group(2) , candidateNouns[1]) and n.group(3) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(Y,X)|' + m.group(1) + '(X,Z)'
                        else:
                            tempelate = m.group(1) + '(X,Y)|' + n.group(1) + '(Z,X)'
                        candidateRelations[candidateNouns[1]].append(tempelate)
                    if (m.group(3) == pronoun and isSame(n.group(3) , candidateNouns[0])) or \
                        (isSame(m.group(3) , candidateNouns[0]) and n.group(3) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(Y,X)|' + m.group(1) + '(Z,X)'
                        else:
                            tempelate = m.group(1) + '(Y,X)|' + n.group(1) + '(Z,X)'
                        candidateRelations[candidateNouns[0]].append(tempelate)
                    if (m.group(3) == pronoun and isSame(n.group(3) , candidateNouns[1])) or \
                        (isSame(m.group(3) , candidateNouns[1]) and n.group(3) == pronoun):
                        if OrigText.find(causal) < OrigText.find(m.group(1)) and OrigText.find(causal) > OrigText.find(n.group(1)):
                            tempelate = n.group(1) + '(Y,X)|' + m.group(1) + '(Z,X)'
                        else:
                            tempelate = m.group(1) + '(Y,X)|' + n.group(1) + '(Z,X)'
                        candidateRelations[candidateNouns[1]].append(tempelate)

            print candidateRelations[candidateNouns[0]],candidateRelations[candidateNouns[1]]

            Scan1 = CalculateKnowledgeValue(candidateRelations[candidateNouns[0]], type2KnowledgeIndex, causal)
            Scan2 = CalculateKnowledgeValue(candidateRelations[candidateNouns[1]], type2KnowledgeIndex, causal)

            print Scan1,Scan2

            if Scan1 == Scan2 == 0:
                continue
            
            if Scan1 >= Scan2:

                print "coref by knowledge: [ ", pronoun, candidateNouns[0], ']'
                print "Winograd Annotated: ", str(DataSets[int(testId)].NAM)
            else:
                print "coref by knowledge: [ ", pronoun, candidateNouns[1], ']'
                print "Winograd Annotated: ", str(DataSets[int(testId)].NAM)

    print (len(TestSetsId), len(LogForNoRelations), len(LogForNoRelationsWithPronoun))


    # TestSentWriter.close()


if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost:9000')
    (DataSets, TrainSetsId, TestSetsId)=readFile.readDataFromFiles(path)
    print ("Enter EXIT to quit")

    # ConstructKnowledgeFromTrainData(DataSets,TrainSetsId,"TrainKnowledge.txt")
    SolvingCoreference(DataSets,TestSetsId, Schema1File, Schema2File)   

    while True:
        text = raw_input(">>> ")
        if len(text) == 0:
            continue
        if text == 'EXIT':
            break
        TestCorefs(text)
        # pass
    # text = "The river destroyed the bridge because the bridge was low"
    # TestCorefs(text)
