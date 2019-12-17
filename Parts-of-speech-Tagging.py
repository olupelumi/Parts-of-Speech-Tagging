################   DO NOT EDIT THESE IMPORTS #################
import math
import random
import numpy
from collections import *

#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
#classes are so cool cause we can create all types of data structures to
class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence



# test_sentence4 = ["The", "items", "were", "both", "in","the", "major", "unions", "."]
# test_sentence5 = ["Conservative", "Republicans", "and", "Democrats", "in", "Congress", "joined", "in", "the", "informal", "Conservative", "Coalition", "."]

# sentence1 = ["Conservative", "Republicans", "and", "Democrats", "in", "Congress", "joined", "in", "the", "Informal", "Conservative", "Coalition", "."]
# sentence2 = ["The", "items", "were", "both", "in", "the", "major", "unions", "."]
# sentence3 = ["Conservative", "Republicans", "and", "Democrats", "in", "Congress", "joined", "in", "the", "informal", "Conservative", "Coalition", "."]
#unique_words = ['Money', 'is', 'nice', '.']
#unique_tags = ['N', 'V', 'A', '.']
#training_data1 = [('Money','N'), ('is','V'),('nice','A'),('.','.')]
#num_tokens = 4
#unique_words2 = ['Kay', 'is', 'awesome', '.']
#unique_tags2 = ['N', 'V', 'A', '.']
#training_data2 = [('Kay','N'), ('is','V'),('awesome','A'),('.','.')]
#num_tokens = 4
def compute_counts(training_data, order):
	"""
	This function takes as in input a list of tuples representing the
	words with their corresponding tags, as well as the order of the markov model.
	It outputs the number of tokens, and dictionaries containing 
	counts of tags and tag sequence pairs and triples, depending on the order of the markov model.
	"""

	#initializing each of the counters
	tokencount = 0
	tagwords = defaultdict(lambda : defaultdict(int))
	tag_count = defaultdict(int)
	previous_tag_sequence = defaultdict(lambda : defaultdict(int))
	two_previous_tag_sequence = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

	#iterate over the size of the dataset so that 
	#items before can be accessed;
	for num in range(len(training_data)):

		tokencount += 1
		tagwords[training_data[num][1]][training_data[num][0]] += 1
		tag_count[training_data[num][1]] += 1

		#If the order of the markov chain is 2, check the frequency of a tag before the given one
		if num != 0:
			if training_data[num-1][1] != ".":
				previous_tag_sequence[training_data[num-1][1]][training_data[num][1]] += 1

		#If the order of the markov chain is 3, check the frequency of the two tags before the given one
		if order == 3:
			if num != 0 and num != 1:
				if training_data[num-2][1] != "." and training_data[num-1][1] != ".":
					two_previous_tag_sequence[training_data[num-2][1]][training_data[num-1][1]][training_data[num][1]] += 1
					
	if order == 2:
		return tokencount, tagwords, tag_count, previous_tag_sequence
	if order == 3:
		return tokencount, tagwords, tag_count, previous_tag_sequence, two_previous_tag_sequence
#Test Cases:
#Count1=compute_counts(training_data1, 2)
#Count2=compute_counts(training_data1, 3)
#Count3=compute_counts(training_data2, 2)
#Count4=compute_counts(training_data2, 3)

def compute_initial_distribution(training_data, order):
	'''
	 Inputs a list of word, POS pairs and returns a dictionary that contains intitial distribution 
	if order is 2, and also if order is 3
	'''
	tag_list=[]

	counter=1
	if order==2: #Bigram model
		init_dist=defaultdict(int)
		for pair in training_data:
			pos=pair[1]
			tag_list.append(pos)
			if counter==1:
				init_dist[pos]+=1
			else:
				if (tag_list[-2]=='.'):
					init_dist[pos]+=1
			counter+=1
		tot_sent=len(init_dist.keys())# total number of sentences
		for tag in init_dist.keys():
			init_dist[tag]=float(init_dist[tag])/tot_sent #Probability that tag appears as the first thing in the sentence
		return(init_dist)

	if order==3:#Trigram
		init_dist=defaultdict(lambda:defaultdict(int))
		for pair in training_data:
			pos=pair[1]
			tag_list.append(pos)
			if counter==2:
				init_dist[tag_list[-2]][tag_list[-1]]+=1
			elif counter>2 and tag_list[-3]=='.' :
				init_dist[tag_list[-2]][tag_list[-1]]+=1
			counter+=1

		tot_senten=len(init_dist.keys())# total number of sentences
		for tag in init_dist.keys():
			for tag2 in init_dist[tag].keys():
				init_dist[tag][tag2]=float(init_dist[tag][tag2])/tot_senten #Probability that tag appears as the first thing in the sentence
		return(init_dist)

#Test Cases:
#In1=compute_initial_distribution(training_data1, 2)
#In2=compute_initial_distribution(training_data1, 3)
#In3=compute_initial_distribution(training_data2, 2)
#In4=compute_initial_distribution(training_data2, 3)

def compute_emission_probabilities(unique_words, unique_tags, W, C):
	"""
	This function takes as an input the list of unique words and tags
	, and returns the emisison probabilities
	of the words given tags as described in the description document.
	"""

	emProb = defaultdict(lambda : defaultdict(int))

	#Iterate over words and tags and check to see if the word-tag pair
	#exists in W, if so, we set the emission probability as 
	#described by the decsciption document.

	for tag in unique_tags:
		for word in unique_words:
			if tag in W:
				if word in W[tag]:
					emProb[tag][word] = float(W[tag][word]) / float(C[tag])

	return emProb

def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
    """
    Computes three lambda values: lambda0, lambda1, lambda2

    Arguments:
        unique_words -- A list of unique words found in the training data
        unique_tags --  A list of unique POS tags found in the training data
        num_tokens -- number of tokens in the training data
        C1 -- C(ti), number of times tag ti appears
        C2 -- C(ti-1, ti), number of times tag seq ti-1, ti appears
        C3 -- C(ti-2, ti-1, ti), number of times tag seq ti-2, ti-1, ti
        order -- the order of the HMM

    Returns:
        a list containing lambda0, lambda1, lambda2 respectively
    """

    lambdas = [0.0, 0.0, 0.0]

    if order == 2:
        # This is for bigram model ti-1, ti with C(ti-1,ti) > 0
        for t1 in unique_tags:
            for t2 in unique_tags:
                if t1 in C2 and t2 in C2[t1] and C2[t1][t2] > 0:
                    alpha0 = 1.0 * (C1[t2] - 1) / num_tokens
                    if C1[t1] > 1:
                        alpha1 = 1.0 * (C2[t1][t2] - 1) / (C1[t1] - 1)
                    else:
                        alpha1 = 0.0
                    i = numpy.argmax([alpha0, alpha1])
                    lambdas[i] = lambdas[i] + C2[t1][t2]
        if (lambdas[0] + lambdas[1]) != 0:
            lambdas = [x / (lambdas[0] + lambdas[1]) for x in lambdas]

    if order == 3:
        # for trigram ti-2, ti-1, ti with C(ti-2, ti-1,ti) > 0
        for t1 in unique_tags:
            for t2 in unique_tags:
                for t3 in unique_tags:
                    if t1 in C3 and t2 in C3[t1] and t3 in C3[t1][t2] and C3[t1][t2][t3] > 0:
                        alpha0 = 1.0 * (C1[t3] - 1) / num_tokens
                        if C1[t2] > 1:
                            alpha1 = 1.0 * (C2[t2][t3] - 1) / (C1[t2] - 1)
                        else:
                            alpha1 = 0.0
                        if C2[t1][t2] > 1:
                            alpha2 = 1.0 * (C3[t1][t2][t3] - 1) / (C2[t1][t2] - 1)
                        else:
                            alpha2 = 0.0
                        i = numpy.argmax([alpha0, alpha1, alpha2])
                        lambdas[i] = lambdas[i] + C3[t1][t2][t3]
        # This is the (lambda0, lambda1, lambda2) / (Lambda0 + lambda1 + lambda2) in  the eqn
        if (lambdas[0] + lambdas[1] + lambdas[2]) != 0:
            lambdas = [x / (lambdas[0] + lambdas[1] + lambdas[2]) for x in lambdas]
    return lambdas

def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	"""
	This function takes the parameters of training data, unique tags, unique words, and boolean use smoothing
	 to make an hmm, and constructs either
	a second or third order model.
	"""

	initDist = compute_initial_distribution(training_data, order)
	counts = compute_counts(training_data, 3)

	num_tokens, W, C1, C2, C3 = counts[0], counts[1], counts[2], counts[3], counts[4]

	emission_matrix = compute_emission_probabilities(unique_words, unique_tags, W, C1)
	firstOrderTransititonMatrix = defaultdict(lambda: defaultdict(int))
	secondOrderTransitionMatrix = defaultdict(lambda: defaultdict(lambda : defaultdict(int)))

	if order == 2:
		lambda0, lambda1, lambda2 = (0,0,0)
		if use_smoothing:
			lambda_list = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
			lambda0, lambda1, lambda2 = lambda_list[0], lambda_list[1], lambda_list[2]

		else:
			lambda0, lambda1, lambda2 = (0,1,0)

		for firstTag in unique_tags:
			for secondTag in unique_tags:
				firstOrderTransititonMatrix[firstTag][secondTag] = (lambda1*C2[firstTag][secondTag]/float(C1[firstTag]) + (lambda0*C1[secondTag]/float(num_tokens)))
		return HMM(order, initDist, emission_matrix, firstOrderTransititonMatrix)

	if order == 3:
		lambda0, lambda1, lambda2 = (0,0,0)
		if use_smoothing:
			lambda_list = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
			lambda0, lambda1, lambda2 = lambda_list[0], lambda_list[1], lambda_list[2]
		else:
			lambda0, lambda1, lambda2 = (0,0,1)

		for firstTag in unique_tags:
			for secondTag in unique_tags:
				for thirdTag in unique_tags:
					if C2[firstTag][secondTag] == 0:
						secondOrderTransitionMatrix[firstTag][secondTag][thirdTag] = (0)+ (lambda1*C2[secondTag][thirdTag]/float(C1[secondTag])) + (lambda0*(C1[thirdTag])/float(num_tokens))
					else:
						secondOrderTransitionMatrix[firstTag][secondTag][thirdTag] = (lambda2*C3[firstTag][secondTag][thirdTag]/float(C2[firstTag][secondTag]))+ (lambda1*C2[secondTag][thirdTag]/float(C1[secondTag])) + (lambda0*(C1[thirdTag])/float(num_tokens))
		return HMM(order, initDist, emission_matrix, secondOrderTransitionMatrix)

#hmm1=build_hmm(training_data1, unique_tags1, unique_words1, 2, True)
#hmm2=build_hmm(training_data2, unique_tags2, unique_words2, 3, False)

def trigram_viterbi(hmm, sentence):
	"""
	This function takes as an input a 2nd order 
	hmm and a sentence. It implements the Viterbi
	algorithm for 2nd order markov chains. The output
	is a list containing tuples, each of which is a word,
	tag pair. 
	"""
	v = defaultdict(lambda:(defaultdict(lambda: defaultdict(int))))
	bp = defaultdict(lambda:(defaultdict(lambda: defaultdict(int))))
	Z = [None] * (len(sentence))
	last_element = len(sentence) - 1

	#Place initial values in v
	for tag in hmm.emission_matrix.keys():
		for tag2 in hmm.emission_matrix.keys():
			pi_log = log_probability(hmm.initial_distribution[tag][tag2])
			max_state_log = log_probability(hmm.emission_matrix[tag][sentence[0]]) + log_probability(hmm.emission_matrix[tag2][sentence[1]])
			v[tag][tag2][1] = pi_log + max_state_log


	#Filling in v and bp
	for num in range(2, len(sentence)):
		for l_prime in hmm.emission_matrix.keys():
			for l in hmm.emission_matrix.keys():

				max_value = float("-inf")
				max_tag = 0

				for l_double_prime in hmm.emission_matrix.keys():

					A_log = log_probability(hmm.transition_matrix[l_double_prime][l_prime][l])
					value_from_v = v[l_double_prime][l_prime][num-1]
					currentScore = value_from_v + A_log

					if currentScore > max_value:
						max_value = currentScore
						max_tag = l_double_prime


				E_log = log_probability(hmm.emission_matrix[l][sentence[num]])

				v[l_prime][l][num] = max_value + E_log
				bp[l_prime][l][num] = max_tag

	index = None
	index2 = None
	max_state = float("-inf")

	#Find the argmax in the last row of v 
	for l_prime in hmm.emission_matrix.keys():
		for l in hmm.emission_matrix.keys():
			if v[l_prime][l][last_element] > max_state:
				max_state = v[l_prime][l][last_element]
				index = l
				index2 = l_prime
	Z[last_element] = tuple([sentence[last_element], index])
	Z[last_element - 1] = tuple([sentence[last_element - 1], index2])

	for num in range((len(sentence)-3), -1, -1): 

		Z[num] = sentence[num], bp[Z[num+1][1]][Z[num+2][1]][num+2]

	return Z
#tp1=trigram_viterbi(hmm, sentence1)
#tp2=trigram_viterbi(hmm, sentence2)
def log_probability(probability):
	"""
	This function converts multiplicative probabiltities
	into their log equivalent, so that the
	probabilities can be added together. 
	"""
	if probability == 0:
		return float("-inf")
	else:
		return math.log(probability)

#log_probability(0.12)
#log_probability(0.67)
def update_hmm(hmm, unique_test_words):
	"""
	This function updates a previously-made hmm with a small
	emission probabiltiy for each word in the test data that is not in the
	training data. More specifically, it creates and returns a new
	hmm with the updated emission matrix values.
	"""
	emission_matrix = hmm.emission_matrix
	seen_words = []

	#Check to see which words have already been seen
	for tag in emission_matrix:
		for word in emission_matrix[tag]:
			seen_words.append(word)
	
	#Update the emission matrix if a words is found
	#that has not already been seen
	for word in unique_test_words:
		if word not in seen_words:
			for tag in emission_matrix:
				for word2 in emission_matrix[tag]:
					emission_matrix[tag][word2] += 0.00001
				emission_matrix[tag][word] = 0.00001
	#Normalize the values in the emission matrix such
	#that they all add up to 1 
	for tag in emission_matrix:
		total = 0.0
		for word in emission_matrix[tag]:
			total += emission_matrix[tag][word]
		for word in emission_matrix[tag]:
			emission_matrix[tag][word] = emission_matrix[tag][word]/ float(total)

	return HMM(hmm.order, hmm.initial_distribution, emission_matrix, hmm.transition_matrix)



def read_text_file(filename):
	"""
	This function reads in untagged sentences/paragraphs of text
	and converts them into a form that can be analyzed 
	by other functions and tagged.
	"""
	data = []
	file = open(filename,"r")
	words = file.read()
	file.close()

	sentence = []
	words = words.split(" ")
	for word in words:
		sentence.append(word)
		if word == ".":
			data.append(sentence)
			sentence = []
	return data



def get_unique(training_data):
	"""
	This function gets the unique tags and words from the slice 
	of the training data it is being run on, and returns
	them in a format similar to read_pos_file.
	"""
	unique_tags = set([])
	unique_words = set([])
	for pair in training_data:
		unique_words.add(pair[0])
		unique_tags.add(pair[1])
	return unique_tags, unique_words

def accuracy(experimental, actual):
	"""
	This function compares each of the tags
	in the data tagged by our algorithms
	to that of the actual tags, and 
	returns an accuracy value.
	"""
	hits = 0

	for num in range(len(experimental)):
		if experimental[num] == actual[num]:
			hits += 1
	return hits / float(len(experimental))

def run_test(filename):
	"""
	This function implements all previously
	wrriten functions in order to tag the testing
	data using each type of markov chain, with and
	without smoothing.
	"""

	# Data partitions
	training_data, what, what2 = read_pos_file("training.txt")
	one_percent_data = training_data[0: int(0.01*len(training_data))]
	five_percent_data = training_data[0: int(0.05*len(training_data))]
	ten_percent_data = training_data[0: int(0.1*len(training_data))]
	twentyFive_percent_data = training_data[0: int(0.25*len(training_data))]
	fifty_percent_data = training_data[0: int(0.5*len(training_data))]
	seventyFive_percent_data = training_data[0: int(0.75*len(training_data))]
	oneHundred_percent_data = training_data

	sentences = read_text_file("testdata_untagged.txt")
	actual_tags = read_pos_file("testdata_tagged.txt")

	#Recalculate unique tags on words for smaller datasets
	unique_tags1, unique_words1 = get_unique(one_percent_data)
	unique_tags2, unique_words2 = get_unique(five_percent_data)
	unique_tags3, unique_words3 = get_unique(ten_percent_data)
	unique_tags4, unique_words4 = get_unique(twentyFive_percent_data)
	unique_tags5, unique_words5 = get_unique(fifty_percent_data)
	unique_tags6, unique_words6 = get_unique(seventyFive_percent_data)
	unique_tags7, unique_words7 = get_unique(oneHundred_percent_data)

	#Set up all possible combinations for experiments
	smoothings = [False, True]
	orders = [2, 3]

	for order in orders:
		for smoothing in smoothings:

			modelOne = build_hmm(one_percent_data, unique_tags1, unique_words1, order, smoothing)
			modelTwo = build_hmm(five_percent_data, unique_tags2, unique_words2, order, smoothing)
			modelThree = build_hmm(ten_percent_data, unique_tags3, unique_words3, order, smoothing)
			modelFour = build_hmm(twentyFive_percent_data, unique_tags4, unique_words4, order, smoothing)
			modelFive = build_hmm(fifty_percent_data, unique_tags5, unique_words5, order, smoothing)
			modelSix = build_hmm(seventyFive_percent_data, unique_tags6, unique_words6, order, smoothing)
			modelSeven = build_hmm(oneHundred_percent_data, unique_tags7, unique_words7, order, smoothing)
			
			model_dict = {1: modelOne, 2: modelTwo, 3: modelThree, 4: modelFour, 5: modelFive, 6: modelSix, 7: modelSeven}
			accuracy_dict = {}

			for num in range (1, 8):
				tagged_selection = []

				for sentence in sentences:

					good_hmm = update_hmm(model_dict[num], sentence)

					if order == 2:
						tagged_selection.extend(bigram_viterbi(good_hmm, sentence))
					if order == 3:
						tagged_selection.extend(trigram_viterbi(good_hmm, sentence))

				accuracy_dict[num] = accuracy(tagged_selection, actual_tags[0])

			print (accuracy_dict)
print (run_test("testdata_untagged.txt"))




# print compute_initial_distribution(training_data1[0], 2)
# print compute_emission_probabilities(training_data1[1], training_data1[2], counts[1], counts[2])
# counts = compute_counts(training_data1[0], 3)
# num_tokens, C1, C2, C3 = counts[0], counts[2], counts[3], counts[4]
# print compute_lambdas(training_data1, num_tokens, C1, C2, C3, 2)
# print compute_lambdas(training_data1[2], num_tokens, C1, C2, C3, 3)
#More test cases:
# hmm = update_hmm(hmm, test_sentence)

# print trigram_viterbi(hmm, test_sentence)

# first_order_hmm = build_hmm(training_data1[0], training_data1[2], training_data1[1], 2, True)

# second_order_hmm = build_hmm(training_data1[0], training_data[2], training_data1[1], 3, False)

# updated_first_order_hmm = update_hmm(first_order_hmm, test_sentence)
# updated_second_order_hmm = update_hmm(second_order_hmm, test_sentence)

# print bigram_viterbi(updated_first_order_hmm, test_sentence)
# print trigram_viterbi(updated_second_order_hmm, test_sentence)




