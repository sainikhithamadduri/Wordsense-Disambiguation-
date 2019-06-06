'''
Introduction :


Authors : Sai Nikhitha Madduri and Merin Joy
Date : 31st October 2018
Word sense disambiguation is a open problem in NLP and ontology, it refers to identification of in which sense a word is
used in a sentence, when the word has several meanings.
This Python program called decision-list.py implements a decision list classifier to perform word sense disambiguation by identifying features from training data. Logarithm of ratio of probabilities of both senses is used to calculate the likelihood of the learned decision rules

Features Used: words, word indices upto 8 position on either directions and created a decision list by learning train data


Results :


Baseline accuracy is 57.14285714285714%
Accuracy after adding learned features is 84.12698412698413%
Confusion matrix is
col_0    phone  product
row_0                  
phone       59        7
product     13       47


Sample my-decision-list:


['-1_word_telephone', 8.45532722030456, 'phone']
['-1_word_access', 7.238404739325079, 'phone']
['-1_word_car', -6.507794640198696, 'product']
['-1_word_end', 6.339850002884625, 'phone']
['1_word_dead', 5.930737337562887, 'phone']
['-1_word_computer', -5.930737337562887, 'product']
['-1_word_came', 5.930737337562887, 'phone']
['-1_word_ps2', -5.930737337562887, 'product']
['-7_word_telephone', 5.930737337562887, 'phone']
['-1_word_gab', 5.672425341971496, 'phone']
['2_word_computers', -5.672425341971496, 'product']
['-2_word_telephone', 5.672425341971496, 'phone']



Instructions to run decision-list.py :


1) Download nltk stop words using nltk.download('stopwords')
2) Run the decision-list.py program in the command prompt as follows:
$ python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt
3) A text file with all decisions will be obtained in my-decision-list.txt and output text file with list of answer
instances and sense id can be found my-line-answers.txt. Also, any output file name can be specified since STDOUT function is used.


Sample Output :

my-line-ansers.txt

'<answer instance="line-n.w8_059:8174:" senseid="phone"/>',
'<answer instance="line-n.w7_098:12684:" senseid="phone"/>',
'<answer instance="line-n.w8_106:13309:" senseid="phone"/>',
'<answer instance="line-n.w9_40:10187:" senseid="phone"/>',
'<answer instance="line-n.w9_16:217:" senseid="196.0"/>',
'<answer instance="line-n.w8_119:16927:" senseid="product"/>',
'<answer instance="line-n.w8_008:13756:" senseid="196.0"/>',
'<answer instance="line-n.w8_041:15186:" senseid="phone"/>'


Algorithm :


Step1: Extract the training and test datasets
Step2: Preprocess the data and remove stopwords and punctuation
Step3: Use conditional frequency distribution for calculating the frequencies of the learned rules from the training dataset
Step4: Use condition probability distribution for calculating the probabilities of the frequencies
Step5: Use logarithm of ratio of probabilities of both senses to calculate the likelihood each decision rule
Step6: Determine the majority sense from training data
Step7: Perform predicitons on the test dataset based on the learned decision rules
Step8: Post the answers from the prediction to STDOUT
Step9: Write the decision rules to a file



'''


import nltk, string, re, math, sys
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LidstoneProbDist
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


# command line arguments for the file sources of training data, testing data, decision list
training_data = sys.argv[1]
testing_data = sys.argv[2]
my_decision_list = sys.argv[3]

# Ambiguous word
root = "line"

# initializing the decision list to an empty list
decision_list = []


# Function to preprocess the textual content
def process_text(text):
    text = text.lower()

    # removing the standard stop word from the text
    stop_words = stopwords.words("english")
    stop_words.extend(string.punctuation)
    
    # treating "lines" and "line" as a single entity
    text = text.replace("lines", "line")
    corpus = [re.sub(r'[\.\,\?\!\'\"\-\_/]','',w) for w in text.split(" ")]
    corpus = [w for w in corpus if w not in stop_words and w != '']
    return corpus


# Function to retrieve a word at a certain index
def get_n_word(n, context):
    root_index = context.index(root)
    n_word_index = root_index + n
    if len(context) > n_word_index and n_word_index >= 0:
        return context[n_word_index]
    else:
        return ""

# Function to add a new "condition" learned from the training data to the decision list
def add_word_cond(cfd, data, n):
    for element in data:
        sense, context = element['sense'], element['text']
        n_word = get_n_word(n, context)
        if n_word != '':
            condition = str(n) + "_word_" + re.sub(r'\_', '', n_word)
            cfd[condition][sense] += 1
    return cfd


# Calculate the logarithm of ratio of sense probabilities
def calculate_log_likelihood(cpd, rule):
    prob = cpd[rule].prob("phone")
    prob_star = cpd[rule].prob("product")
    div = prob / prob_star
    if div == 0:
        return 0
    else:
        return math.log(div, 2)


# checking whether the rule is satisfied in a given context
def check_rule(context, rule):
    rule_scope, rule_type, rule_feature = rule.split("_")
    rule_scope = int(rule_scope)
    
    return get_n_word(rule_scope, context) == rule_feature
        
# Function to predict the sense on test data
def predict(context, majority_label):
    for rule in decision_list:
        if check_rule(context, rule[0]):
            if rule[1] > 0:
                return ("phone", context, rule[0])
            elif rule[1] < 0:
                return ("product", context, rule[0])
    return (majority_label, context, "default")


# Extracting the textual content from training data through XML parsing
with open(training_data, 'r') as data:
    soup = BeautifulSoup(data, 'html.parser')
train_data = []
for instance in soup.find_all('instance'):
    sntnc = dict()
    sntnc['id'] = instance['id']
    sntnc['sense'] = instance.answer['senseid']
    text = ""
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
    sntnc['text'] = process_text(text)
    train_data.append(sntnc)

# Use conditional frequency distribution to add learned rules to the decision list
cfd = ConditionalFreqDist()
cfd = add_word_cond(cfd, train_data, 1)
cfd = add_word_cond(cfd, train_data, -1)
cfd = add_word_cond(cfd, train_data, 2)
cfd = add_word_cond(cfd, train_data, -2)
cfd = add_word_cond(cfd, train_data, 3)
cfd = add_word_cond(cfd, train_data, -3)
cfd = add_word_cond(cfd, train_data, 4)
cfd = add_word_cond(cfd, train_data, -4)
cfd = add_word_cond(cfd, train_data, 5)
cfd = add_word_cond(cfd, train_data, -5)
cfd = add_word_cond(cfd, train_data, 6)
cfd = add_word_cond(cfd, train_data, -6)
cfd = add_word_cond(cfd, train_data, 7)
cfd = add_word_cond(cfd, train_data, -7)
cfd = add_word_cond(cfd, train_data, 8)
cfd = add_word_cond(cfd, train_data, -8)


# Instantiating Condition probability distribution to calculate the probabilities of the frequencies recorded above
cpd = ConditionalProbDist(cfd, LidstoneProbDist, 0.1)

# storing the learned rules into the decision list
for rule in cpd.conditions():

    likelihood = calculate_log_likelihood(cpd, rule)
    decision_list.append([rule, likelihood, "phone" if likelihood > 0 else "product"])
    
    decision_list.sort(key=lambda rule: math.fabs(rule[1]), reverse=True)


# extracting the test data through XML parsing
with open(testing_data, 'r') as data:
    test_soup = BeautifulSoup(data, 'html.parser')

test_data = []
for instance in test_soup('instance'):
    sntnc = dict()
    sntnc['id'] = instance['id']
    text = ""
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
    sntnc['text'] = process_text(text)
    test_data.append(sntnc)

# Calculating the frequencies of each senses
senseA, senseB = 0.0, 0.0
for element in train_data:
    if element['sense'] == "phone":
        senseA += 1.0
    elif element['sense'] == 'product':
        senseB += 1.0
    else:
        print("warning no match")

# Calculating the majority sense
majority_sense = "phone" if senseA > senseB else "product"

# Performing the predictions
predictions = []
for element in test_data:
    pred, _, r = predict(element['text'], majority_sense)
    id1 = element['id']
    predictions.append(f'<answer instance="{id1}" senseid="{pred}"/>')
    print(f'<answer instance="{id1}" senseid="{pred}"/>')


# Storing the decision list into a file
with open(my_decision_list, 'w') as output:  
    for listitem in decision_list:
        output.write('%s\n' % listitem)

