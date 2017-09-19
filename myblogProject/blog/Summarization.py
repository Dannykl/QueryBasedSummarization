# class Summarization:
	# def main(a):
	# 	return a + " peopleeeee"
	# if __name__ == "__main__":
	# 	main(a)


"""
REQUIRMENTS IN ORDER TO RUN THE PROGRAM 
- install python
- install NLTK and download stopword corpus,wordnet,pos_tag
- install scikit-fuzzy
- collections
- operator, glob, os, re, math,string

"""
import nltk,glob,os,re,math,string,operator,array
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from operator import itemgetter
import skfuzzy as fuzz
from skfuzzy import control
from django.conf import settings

class Summarization:
	global _reading,dataProcessing,retrieveTheRelevantDocs,queryFormulation,removingStopWords
	global listToString,creatingVector,sentenceAnalysis,idf,display,howManydigitsInSentence,sft
	global howManydigitsInSentence,determineCosine,fuzzy,determineTag
	
	def _reading():
		#bpoild dataset
		#file = open('Data/InputDocs/bpoil_bbc/bpfolders.txt', "r")
		
		path = os.path.join(settings.BASE_DIR, 'blog/Data/InputDocs/bpoil_bbc/bpfolders.txt')
		file = open(path,'r')
		file = file.read().split(',')
		f = []
		for w in file:
			f.append(w.strip('\n'))
		
		val = ''
		count = 0
		containsAllDocs = []
		for x in f:
			val = x.lstrip(' ')
			fi = glob.glob('blog/Data/InputDocs/bpoil_bbc/%s/*.txt'%val)[0]
			
			file2 = open('%s'%(fi), "r")
			ff = file2.read()
			count +=1
			containsAllDocs.append(ff)
		return containsAllDocs

	def dataProcessing(doc):
		#stemmer can be used to stem the term 
	    stemmer = nltk.stem.porter.PorterStemmer()
	    stemmed = []
	    r = []
	    
	    #iterate through each document
	    for each in doc:
	    	#change the document to a collection of tokens
	        tokeniser = nltk.word_tokenize(each.lower())
	        #iterate through the tokens and filter out the stop words
	        filtered = [term for term in tokeniser if removingStopWords(term)]
	        #apply stem for remainingtokens
	        stemmed.append([stemmer.stem(item) for item in filtered])

	    #return the stemmed tokens
	    return stemmed

	#this method takes all the documents and user query
	def retrieveTheRelevantDocs(allDocs,query):
		#call ataProcessing method by passing all the documents
		stemmedDoc = dataProcessing(allDocs)
		
		#declare empty list
		frequencyOfTermsForAll = []
		
		#iterate through the stemmed document
		for eachStemDoc in stemmedDoc:
			#determine the frequency of the term
			frequencyOfTerms = Counter(eachStemDoc)
			#determine the length of the document
			lengthOfDoc = len(frequencyOfTerms)
			#append the frequenc of the term into the list
			frequencyOfTermsForAll.append(frequencyOfTerms)
		
		#PUT THE DOCUMENT WITH ITS TERM FREQUENCY AND ROW DOCUMENT IN THE SAME LIST
		see = zip(frequencyOfTermsForAll, allDocs)
		
		rawAndProbability =[]

		for termOcc,row in see:
			lenOfDoc = len(termOcc)
			probabilityOfTerm = []
			#ITERATE THROUGH THE DOCUMENTS THAT ITS TERMS WITH THEIR FREQUENCY
			for (term,termFreq) in termOcc.items():
				#determine the term probability in the document by simply diving 
				#the term frequency by the lengthe of the document
				termProbInDoc = float(termFreq)/lenOfDoc
				probabilityOfTerm.append((term,termProbInDoc))

			rawAndProbability.append((probabilityOfTerm,row))
		
		probabilityOfDoc = {}
		for termPro,actual in rawAndProbability:
			test = {}
			for eachTerm in query:
				if removingStopWords(eachTerm):
					product = 0
					
					#iterate through the term and its probability
					# check the term in query and term in document are equal
					# assign probability value for query term if the condition is true
					for term,proba in termPro:
						if (eachTerm.lower() == term.lower()) :
							product = proba
							#ASSIGN THE QUERY TERM AND THE PROBABILITY AS KEY AND VALUE INTO TEST DICT
							test[eachTerm] = proba
			
			#check if the dictionary is not empty
			if len(test) != 0:
				#determine the product
				pOfQueryInDoc = list(np.prod(test.values()))
				pOfQueryInDoc = pOfQueryInDoc * 100
				probabilityOfDoc[actual]=pOfQueryInDoc
		topK = int (((len(probabilityOfDoc) * 90)/100))
		print("top k is ",type(topK))

		#sort the document in descending by their probability and the topK documents will be added in the rDocuments 
		rDocuments = dict(sorted(probabilityOfDoc.items(), key=operator.itemgetter(1), reverse=True)[:topK])
		return ((rDocuments.keys()))
			
	def queryFormulation(query):
		container = []
		#iterate through the query terms
		for each in query:
			#check the  query term is not in stopword list by calling the removingStopWords
			if removingStopWords(each):
				#gets the synonym from word net 
				similarWords = [st.lemma_names()[0] for st in wn.synsets(each)]
				#add the unique synonym words into the set
				addingWords = set(similarWords)
				j=0
				for i in addingWords:
					if(j < 5): 

						container.append((i))
					else:
						break
					j +=1
				if each not in container:
					container.append(each)
		return (listToString(container).split(" "))

	def removingStopWords(word):
		if word not in stopwords.words('english'):
			return True

	def listToString(container):
		sentences = ''
		sentences = sentences + ' '.join(container)
		return sentences

	def creatingVector(sequenceOfWords):
	    terms = re.compile(r'\w+').findall(sequenceOfWords) 
	    stemmer = nltk.stem.porter.PorterStemmer()
	    stemmed = [stemmer.stem(i) for i in terms] 
	       
	    tf = Counter(stemmed)
	    sentenceLength = len(tf)
	    for term,fre in tf.items():
	    	#tf[term]/=float(sentenceLength)
	    	tf[term] = math.log(fre+1,10)
	    return (tf,set(stemmed))

	def sentenceAnalysis(retrievedDocs,query):
		tfQuery,stemmedQuery= creatingVector(query)
		allSent = []
		storeRowSentences = []

		#N is number of sentence in corpus
		N=0 
		for eachDoc in retrievedDocs:
			positionOfSentence = 0
			allSent = []
			for eachSentence in eachDoc.split("."):
				N+=1
				positionOfSentence +=1
				tfS,stemmedSen=creatingVector(eachSentence)
				cosineValue = determineCosine(tfQuery,tfS)
				#print(cosineValue)
				allSent.append(tfS)
				storeRowSentences.append((eachSentence,positionOfSentence,cosineValue))
		tfAndsft = sft(allSent,tfQuery)

		#ITERATE THROUGH THE SENTENCE AND QUERY
		scoreForEachSentence = []
		for eachS in allSent:
			summation = 0
			for word,v in eachS.items():
				for term,value in tfAndsft.items():
					if word==term:
						#N is the total number of sentences in the collection.
						#value[1] is the number of sentence that contain query term
						#10 is the logarithm base
						#value[0] frequecy of query term, v is frequency of sentence term
						#log = math.log((N+1)/(0.5 * value[1]),10)
						idfScore = idf(N,value[1])
						summation = value[0]*v *(idfScore)
			scoreForEachSentence.append(summation)
		
		#ASSIGN THE SCORE TO THE ASSOCIATED ROW SENTENCES
		sentenceWithScore = zip(storeRowSentences, scoreForEachSentence)
		#print(storeRowSentences)
		allFeatures = []
		longSentence = 0
		shortSentence = 1000
		for s in sentenceWithScore:
			
			sentence = s[0][0].split(" ")
			nnp,noun,verb  = determineTag(s[0][0])
			lengthOfSentence = len(sentence)
			digit = howManydigitsInSentence(sentence)

			#THE ACTUALSENTENCE,position,cosine,relevantScore,properN,noun,vern,length,howManyDigitalValueinSen
			sen = [s[0][0],s[0][1],s[0][2],s[1],nnp,noun,verb,lengthOfSentence,digit]
			allFeatures.append(sen)
			sen = []

		return(allFeatures)
	def idf(N,n):
		base = 10
		idfScore =  math.log((N+1)/(0.5 * n),base)
		return idfScore

	def display(extractedSentence):
		s = " "
		for ss in extractedSentence:
			s += ss[0] + "."
		return s
		
	def howManydigitsInSentence(sentence):
		count = 0
		for word in sentence:
			if word.isdigit():
				count+=1
		return count

	#sft determines number of sentences that contains query term(t)
	def sft(allSent,tfQuery):
		lenSentencesConT = 0
		for term,fre in tfQuery.items():
			count =0 
			for content in allSent:
				for word in content:
					if word == term:
						count +=1
						tfQuery[term] = fre,count
		for x,y in tfQuery.items():
			
			tu =()
			if type(y) == float:
				tu = (y,0)
				tfQuery[x]=tu
		return (tfQuery)

	def determineTag(sentence):
		taggedSentence = pos_tag(sentence.split())
		NNP = 0
		NN = 0
		VBD = 0
		for word,tag in taggedSentence:
			if tag == 'NNP':
				NNP +=1 
			if tag == 'NN':
				NN += 1
			if tag == 'VBD':
				VBD +=1
			#print(word,tag)
		#pNouns =[word for word,tag in taggedSentence if tag == 'NNP']
		NNP /=float(10)
		NN /=float(10)
		VBD /=float(10)

		return NNP,NN,VBD
	def determineCosine(query, sentence):
	     #inserSec holds the intersection words from the query and sentences 
	     interSec = set(query.keys()).intersection(sentence.keys())
	     
	     #multiples the frequency value in query with the frequency value in sentence & sum up
	     num = sum([query[x] * sentence[x] for x in interSec])

	     #square the value of query term and sum up and do the same thing for sentence query
	     #the result from both will be multiplied
	     denom = math.sqrt(sum([query[x]**2 for x in query.keys()])) * \
	     math.sqrt(sum([sentence[x]**2 for x in sentence.keys()]))

	     if not denom:
	        return 0.0
	     else:
	        return float(num) / denom
	def fuzzy(sentencePosition,sentenceSimilarityToTheUserQuery,sentenceRelevance,sentenceContainsProperNoun,sentenceContainsNoun,sentenceContainsVerb,sentenceLength,sentenceContainsDigitalData):
		
		# the Antecedent objects hold universe variables and membership
		isSentencePosition = control.Antecedent(np.arange(0,1.1,.1), 'isSentencePosition')
		sentenceSimilarityToTheQuery = control.Antecedent(np.arange(0,1.1,.1), 'sentenceSimilarityToTheQuery')
		sentenceRelevanceToTheQuery = control.Antecedent(np.arange(0,1.1,.1), 'sentenceRelevanceToTheQuery')
		sentenceContainsSpecificNames = control.Antecedent(np.arange(0,1.1,.1), 'sentenceContainsSpecificNames')
		sentenceContainsGeneralNames = control.Antecedent(np.arange(0,1.1,.1), 'sentenceContainsGeneralNames')
		IsSentenceComplete = control.Antecedent(np.arange(0,1.1,.1), 'IsSentenceComplete')
		sentenceIsAdequateLength = control.Antecedent(np.arange(0,1.1,.1), 'sentenceIsAdequateLength')
		sentenceContainsDigitals = control.Antecedent(np.arange(0,1.1,.1), 'sentenceContainsDigitals')

		# the consequent object holds universe variables and membership
		sentenceImportanceIs = control.Consequent(np.arange(0,1.1,.1), 'sentenceImportanceIs')

		'''Auto-membership function population is possible with .automf(3, 5, or 7)
		Here automf(5) is used which are:good,decent,average , mediocre and poor
		'''
		isSentencePosition.automf(5)
		sentenceSimilarityToTheQuery.automf(5)
		sentenceRelevanceToTheQuery.automf(5)
		sentenceContainsSpecificNames.automf(5)
		sentenceContainsGeneralNames.automf(5)
		IsSentenceComplete.automf(5)
		sentenceIsAdequateLength.automf(5)
		sentenceContainsDigitals.automf(5)

		#describe the antecedent,sentenceImportanceIs, with 5 terms - each term has  
		sentenceImportanceIs['veryLow'] = fuzz.trimf(sentenceImportanceIs.universe, [-0.1, 0.05, 0.1])
		sentenceImportanceIs['low'] = fuzz.trimf(sentenceImportanceIs.universe, [0.11, 0.20, 0.30])
		sentenceImportanceIs['average'] = fuzz.trimf(sentenceImportanceIs.universe, [0.31, 0.40, 0.50])
		sentenceImportanceIs['high'] = fuzz.trimf(sentenceImportanceIs.universe, [0.51, 0.60, 0.70])
		sentenceImportanceIs['veryHigh'] = fuzz.trimf(sentenceImportanceIs.universe, [.71, .85, 1])

		'''
		-0.1 - 0.10 ==> poor
		0.11 - 0.30 ==> mediocre
		0.31 - 0.50 ==> average ,
		0.51 - 0.70 ==> decent, 
		0.71 - 1 ==> good,    
		'''
		
		
		
		#if(sentenceSimilarityToTheQuery['good'] OR sentenceSimilarityToTheQuery['decent']) AND 
		#sentenceContainsSpecificNames['good']OR sentenceContainsGeneralNames['good'] OR 
		#IsSentenceComplete['good'] THEN sentenceImportanceIs['veryHigh'])
		rule1 = control.Rule((sentenceSimilarityToTheQuery['good']|sentenceSimilarityToTheQuery['decent']) & 
			sentenceContainsSpecificNames['good']|sentenceContainsGeneralNames['good']|IsSentenceComplete['good'], 
			sentenceImportanceIs['veryHigh'])
		
		rule2 = control.Rule((sentenceSimilarityToTheQuery['good']|sentenceSimilarityToTheQuery['decent'])&
			(sentenceRelevanceToTheQuery['good']|sentenceRelevanceToTheQuery['decent']),
			sentenceImportanceIs['veryHigh'])

		rule3 = control.Rule((sentenceRelevanceToTheQuery['good']|sentenceRelevanceToTheQuery['decent']) & 
			sentenceContainsSpecificNames['good']|
			sentenceIsAdequateLength['average']|sentenceIsAdequateLength['mediocre'],
			sentenceImportanceIs['veryHigh'])

		rule4 = control.Rule(isSentencePosition['poor']& (sentenceSimilarityToTheQuery['mediocre']|
			sentenceSimilarityToTheQuery['average'])&
			(sentenceRelevanceToTheQuery['average']|sentenceRelevanceToTheQuery['mediocre']) & 
			sentenceContainsSpecificNames['good'] & 
			sentenceContainsGeneralNames['good'] & IsSentenceComplete['good'] | (sentenceContainsDigitals['good']), 
			sentenceImportanceIs['high'])

		rule5 = control.Rule((sentenceSimilarityToTheQuery['average']|sentenceSimilarityToTheQuery['mediocre'])&(sentenceRelevanceToTheQuery['average']|sentenceRelevanceToTheQuery['mediocre']),
			sentenceImportanceIs['high'])


		rule6 = control.Rule((sentenceRelevanceToTheQuery['average']|sentenceRelevanceToTheQuery['mediocre'])&(sentenceSimilarityToTheQuery['poor']|sentenceSimilarityToTheQuery['mediocre']),
			sentenceImportanceIs['average'])

		rule7 = control.Rule((sentenceSimilarityToTheQuery['poor']|sentenceRelevanceToTheQuery['poor'])&sentenceContainsGeneralNames['good'] & IsSentenceComplete['good'], 
			sentenceImportanceIs['average'])
		rule8 = control.Rule((sentenceIsAdequateLength['average']|sentenceIsAdequateLength['mediocre'])&sentenceContainsGeneralNames['good']&sentenceContainsSpecificNames['good']&IsSentenceComplete['good'],
			sentenceImportanceIs['average'])

		rule9 = control.Rule(sentenceSimilarityToTheQuery['mediocre']&sentenceRelevanceToTheQuery['mediocre'] & sentenceContainsSpecificNames['poor'] & sentenceContainsGeneralNames['poor'] & IsSentenceComplete['poor'],
		 	sentenceImportanceIs['low'])
		rule10 = control.Rule(sentenceSimilarityToTheQuery['poor']|sentenceRelevanceToTheQuery['poor'] & sentenceContainsSpecificNames['poor'] & sentenceContainsGeneralNames['good']& IsSentenceComplete['poor'], 
			sentenceImportanceIs['low'])

		rule11 = control.Rule(sentenceSimilarityToTheQuery['poor']&sentenceRelevanceToTheQuery['poor'] & sentenceContainsSpecificNames['poor'] & sentenceContainsGeneralNames['poor'], 
			sentenceImportanceIs['veryLow'])

		
		senControl = control.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11])
		sentenceImportanceIs = control.ControlSystemSimulation(senControl)

		sentenceImportanceIs.input['isSentencePosition'] = sentencePosition
		sentenceImportanceIs.input['sentenceSimilarityToTheQuery'] = sentenceSimilarityToTheUserQuery
		sentenceImportanceIs.input['sentenceRelevanceToTheQuery'] = sentenceRelevance
		sentenceImportanceIs.input['sentenceContainsSpecificNames'] = sentenceContainsProperNoun
		sentenceImportanceIs.input['sentenceContainsGeneralNames'] = sentenceContainsNoun
		sentenceImportanceIs.input['IsSentenceComplete'] = sentenceContainsVerb
		sentenceImportanceIs.input['sentenceIsAdequateLength'] = sentenceLength
		sentenceImportanceIs.input['sentenceContainsDigitals'] = sentenceContainsDigitalData
		sentenceImportanceIs.compute()

		return sentenceImportanceIs.output['sentenceImportanceIs']

	def main(userQuery):
		#storageLoca = "../Evaluation/evaluation-master/ROUGE-RELEASE-1.5.5/summaries/system/bp.txt"
		#writeSummaryOnFile = open(storageLoca, "w")
		allDocs  = _reading()
		query = userQuery.split(" ")
		#query = "when the BP oil spill was stopped".split(" ")
		reformQ = queryFormulation(query)

		que = listToString(reformQ)


		retrievedDocs = []

		
		#THIS SECTION SHOULD BE COMMMENTED DURING EVALUATION AS THE RETRIEVAL PART IS NOT USED
		if len(queryFormulation(query)) > 1 :
			retrievedDocs = retrieveTheRelevantDocs(allDocs,query)
		
		#print("the length of the retrieved doc are ",len(retrievedDocs))
		labelledSentences = sentenceAnalysis(retrievedDocs,que)

		sentenceAndFinalScore = []
		#print(len(labelledSentences))
		
		
		for sentence in labelledSentences:
			position = sentence[1] /100
			length = sentence[7] /100
			#THE SET OF INPUTS SHOULD MEET ATLEAST ONE RULE
			finalScore = fuzzy(position,sentence[2],sentence[3],sentence[4],sentence[5],sentence[6],length,sentence[8])
		
			sentenceAndFinalScore.append((sentence[0],finalScore))
		#print("is ",len(sentenceAndFinalScore))
		#COMMENT THE IF THEN STATEMENT DURING EVALUATION
		topK = 50
		if len(sentenceAndFinalScore) < 50:
			#topK is 20% of the total number of sentences
			topK = len(sentenceAndFinalScore)
			
			#topK = (len(sentenceAndFinalScore) * 20)/100
			#sort the sentence by the value and selects the topK which is the compressed value
			extractedSentence = sorted(sentenceAndFinalScore, key=itemgetter(1), reverse=True)[:topK]
			return (display(extractedSentence))
			#writeSummaryOnFile.write(display(extractedSentence))
			#writeSummaryOnFile.close()
		else:
			print(topK)
			extractedSentence = sorted(sentenceAndFinalScore, key=itemgetter(1), reverse=True)[:topK]
			return (display(extractedSentence))
			#writeSummaryOnFile.write(display(extractedSentence))
			#writeSummaryOnFile.close()
		

	if __name__ == '__main__':
		main(query)

