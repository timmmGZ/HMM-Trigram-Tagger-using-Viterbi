from nltk.corpus import conll2000
from collections import  defaultdict
from nltk import  trigrams
from itertools import chain


def myRandomAlgorithm(testSent, testTags, trainWords, trainTags, eModel, tModel, wrongCount):
    status = ['<s>', '<s>']
    t = dict.fromkeys(trainTags, 1)
    for i in range(2, len(testSent) - 1):
        if testSent[i] in trainWords:
            bestProb = dict.fromkeys(trainTags, 0)
            for x in eModel:
                for y in eModel:
                    bestProb[x] += tModel[(status[i - 2], status[i - 1])][y] * eModel[x][testSent[i]] * t[y]
            status.append(max(bestProb, key=lambda x: bestProb[x]))
            t = bestProb
        else:
            status.append('NN')
        if status[i] != testTags[i]:
            wrongCount[0] += 1
    status.append('<\s>')
    return status


def viterbi(testSent, testTags, trainWords, trainTags, eModel, tModel, wrongCount):
    chain = {}
    for state in eModel:
        chain[state] = ['<s>', '<s>']
    lastProbs = dict.fromkeys(trainTags, 1)
    for i in range(2, len(testSent)-1):
        tmpChain = {}
        for state in eModel:
            tmpChain[state] =  ['<s>', '<s>']
        if testSent[i] in trainWords:
            bestProb = dict.fromkeys(trainTags, 0)  
            for x in eModel:         
                for y in tModel:
                    if  tModel[y][x] * eModel[x][testSent[i]] * lastProbs[y[1]] > bestProb[x]:
                        bestProb[x] = tModel[y][x] * eModel[x][testSent[i]] * lastProbs[y[1]] 
                        tmpChain[x] = list(chain[y[1]])
                        tmpChain[x].append(x)
            lastProbs = bestProb
            chain = tmpChain 
        else:
            for state in eModel:
                chain[state].append('NN')
    bestStatus = ""
    pro = 0
    for x in eModel:
        if lastProbs[x] > pro:
            pro = lastProbs[x]
            bestStatus = x 
            
    for i in range(2, len(testSent) - 1):
        if  chain[bestStatus][i] != testTags[i]:
            wrongCount[0] += 1
            
    chain[bestStatus].append('<\s>')
    return chain[bestStatus]


def HMMAccuracy(testSents, trainWords, trainTags, eModel, tModel, wrongCount):
    sentNum = 0
    totalSentNum = len(testSents)
    total = len(list(chain.from_iterable(testSents)))
    for sentence in testSents:
        sentNum += 1
        refreshPrint((str(sentNum) + '/' + str(totalSentNum))+", accuracy: "+str((total - wrongCount[0]) / total*100)+"%")
        testTags = ['<s>', '<s>'] + [k[1] for k in sentence] + ['<\s>']
        testSent = ['<s>', '<s>'] + [k[0] for k in sentence] + ['<\s>']
        viterbi(testSent, testTags, trainWords, trainTags, eModel, tModel, wrongCount)
    return (total - wrongCount[0]) / total


def kFoldCV(fold):
    if fold <= 1:
        print("please enter fold >1!")
        return
    else:
        l = int(len(conll2000.tagged_sents()) / fold)
        for i in range(fold):
            refreshPrint(str(fold) + ' fold cross-validation: preparing for ' + str(i + 1) + ' loop\'s tagger model...')  
            left = l * i
            right = left + l
            testSents = conll2000.tagged_sents()[left:right]
            trainSents = conll2000.tagged_sents()[:left] + conll2000.tagged_sents()[right:]
            trainTags = set(k[1] for k in chain.from_iterable(trainSents)) | {'<s>', '<\s>'}
            trainWords = set([a for a, b in set(chain.from_iterable(trainSents))])
            allTrainWords = list(chain.from_iterable(trainSents))
            ##################trainsition Probablilities       
            tModel = defaultdict(lambda: defaultdict(lambda: 0))
            
            for x in trainTags:
                for y in trainTags:
                    for z in trainTags:
                        tModel[(x, y)][z] = 1
            
            for sentence in trainSents:
                sentence = [('', '<s>'), ('', '<s>')] + sentence + [('', '<\s>')]
                for x, y, z in trigrams([k[1] for k in sentence]):
                    tModel[(x, y)][z] += 1
            
            for xy in tModel:
                totalCount = float(sum(tModel[xy].values()))
                for z in tModel[xy]:
                    tModel[xy][z] /= totalCount
            
            ########################emission Probablities
            eModel = defaultdict(lambda: defaultdict(lambda: 0))
            
            for x in trainTags:
                for z in trainWords:
                    eModel[x][z] = 1
            
            for z, x in allTrainWords:
                eModel[x][z] += 1
            
            for x in eModel:
                total_count = float(sum(eModel[x].values()))
                for z in eModel[x]:
                    eModel[x][z] /= total_count
            
            wrongCount = [0]
            refreshPrint('                                                                                                              ')
            print(str(i + 1) + ' loop accuracy: ' + str(HMMAccuracy(testSents, trainWords, trainTags, eModel, tModel, wrongCount)*100) + '%')
    

def refreshPrint(s):
    print(s, end='')
    print('\r', end='')

    
kFoldCV(10)
