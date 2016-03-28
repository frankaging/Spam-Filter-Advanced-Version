############################################################
# CIS 521: Homework 6
############################################################

student_name = "Zhengxuan Wu"

############################################################
# Imports
############################################################


# Include your imports here, if any are used.
import email
import math
import os
import Queue
import heapq
from collections import Counter
import string
exclude = set(string.punctuation)


############################################################
# Section 2: Spam Filter
############################################################



def delete_stop(li):
    resul = []
    stop_li = set(['HTML','FONT','BR','SIZE','CONTENT','Content','content','html','BODY','>','<','><','<>','=','</','-','TITLE','"','HTTP','EQUIV','EN','="','">','"=','<"','META','DOCTYPE','PUBLIC','DTD','NAME',"'",'</','/>','/<','>/','W3C','MS','Generator','Server','P', 'China', 'India','Exchange','version','Type','U','S','Germany','Japan','Korea','Peru','Russian','Spain','Sweden','Australia','Canada','Brazil','the','a','and'])
    for element in li:
        if element not in stop_li and element.lower() not in stop_li:
            resul.append(element)
    return resul

def load_tokens(email_path):
    f = open(email_path)
    m = email.message_from_file(f)
    token_list = []
    no_punc_token = []
    
    for line in email.iterators.body_line_iterator(m):
        line = line.strip()
        li = line.split()
        for element in li:
            ##########################################
            temp_none_punc = []
            if len(element) <= 45:
                for char in element:
                    if char.isdigit() or char.isalpha():
                        temp_none_punc.append(char)
                    else:
                        t="".join(temp_none_punc)
                        if t != "":
                            no_punc_token.append(t)
                        temp_none_punc = []
                t1 = "".join(temp_none_punc)
                if t1 != "":
                    no_punc_token.append(t1)
                ##########################################
    return (token_list,delete_stop(no_punc_token))

#spam_dir = "data/dev/spam/"
#print load_tokens(spam_dir+"dev283")[0]
#load_tokens(spam_dir+"dev283")[1]

def bigram_token(li):
    bigram_tokens = []
    if len(li) > 1:
        for i in xrange(1,len(li)):
            temp = " ".join([li[i-1],li[i]])
            bigram_tokens.append(temp)
        return bigram_tokens
    else:
        return []

def cap_num(s):
    num = 0
    for letters in s:
        if letters.isupper():
            num +=1
    return num

def only_al(s):
    for char in s:
        if not char.isalpha():
            return False
    return True

def log_probs(email_paths):
    # 9, 15
    smoothing_uni = 1e-9
    smoothing_bi = 1e-15
    smoothing_other = 1
    
    freq = dict()
    bigram_freq = dict()
    freq["upper0"] = freq["upper1"] = freq["upper_all"] = freq["upper_rand"] = freq["illegal_element"] = freq["legal_element"] = 0
    for path in email_paths:
        tokens = []
        temp = load_tokens(path)
        tokens = temp[1]
        bigrams = bigram_token(tokens)
        tokens.extend(bigrams)
        for element in set(tokens):
            freq[element] = freq.get(element,0) + 1
            if len(element.split()) == 1:
                
                # capitalization
                if len(element) > 2:
                    if cap_num(element) == 0:
                        freq["upper0"] = freq.get("upper0") + 1
                    elif cap_num(element) == len(element):
                        freq["upper_all"] = freq.get("upper_all") + 1
                    elif element[0].isupper() and cap_num(element) != len(element):
                        freq["upper1"] = freq.get("upper1") + 1
                    else:
                        freq["upper_rand"] = freq.get("upper_rand") + 1

                # length
                if len(element) < 6:
                    freq["len_bin_1"] = freq.get("len_bin_1",0) + 1
                elif len(element) >=6 and len(element) < 15:
                    freq["len_bin_3"] = freq.get("len_bin_3",0) + 1
                else:
                    freq["len_bin_4"] = freq.get("len_bin_4",0) + 1

                # num or not
#                if element.isdigit():
#                    freq["digit_only"] = freq.get("digit_only",0) + 1
#                else:
#                    freq["digit_mix"] = freq.get("digit_mix",0) + 1

#    #   second round training
#    sum = 0
#    for path in email_paths:
#        tokens = []
#        temp = load_tokens(path)
#        tokens = temp[1]
#        bigrams = bigram_token(tokens)
#        tokens.extend(bigrams)
#        ind = 0
#        for element in set(tokens):
#            if  freq[element] < 10:
#                freq["illegal_element"] = freq.get("illegal_element") + 1
#                ind +=1
#            else:
#                freq["legal_element"] = freq.get("legal_element") + 1
#    
#        if len(tokens) != 0:
#            sum += ind*1.0/len(tokens)

        # print sum/1000.0

    # feature selection to reduce time
    li_keys = freq.keys();
    for key in li_keys:
        if len(key.split()) == 2 and freq[key] < 2:
            del freq[key]
    #print freq["len_bin_1"], freq["len_bin_3"], freq["len_bin_4"]
    #print freq["upper0"], freq["upper1"], freq["upper_all"], freq["upper_rand"]
    total_count = 0
    vocab_length = 0
    for i in freq:
        total_count += freq[i]
        vocab_length += 1
    # get the frequency table
    for key in freq.keys():
        if key in ["upper0", "upper1", "upper_all", "upper_rand", "len_bin_1", "len_bin_2", "len_bin_3", "len_bin_4", "digit_only", "digit_mix"]:
            freq[key] = math.log((freq.get(key,0)+smoothing_other)/(total_count + (smoothing_other*(vocab_length+2.0))))
        elif len(key.split()) == 2:
            freq[key] = math.log((freq[key]+smoothing_bi)/(total_count + (smoothing_bi*(vocab_length+2.0))))
        else:
            freq[key] = math.log((freq[key]+smoothing_uni)/(total_count + (smoothing_uni*(vocab_length+2.0))))
    freq["<UNK>_uni"] = math.log((smoothing_uni/(total_count + (smoothing_uni*(vocab_length+2.0)))))
    freq["<UNK>_bi"] = math.log((smoothing_bi/(total_count + (smoothing_bi*(vocab_length+2.0)))))


#print freq
#    print bigram_freq
    return freq


#paths = ["data/train/ham/ham%d" % i for i in range(1, 255)]
#p = log_probs(paths)
#paths = ["data/train/spam/spam%d" % i for i in range(1, 255)]
#p = log_probs(paths)
#print a

class SpamFilter(object):

    # Note that the initialization signature here is slightly different than the
    # one in the previous homework. In particular, any smoothing parameters used
    # by your model will have to be hard-coded in.

    def __init__(self, spam_dir, ham_dir):
        _spam = [item for item in os.listdir(spam_dir) if os.path.isfile(os.path.join(spam_dir, item))]
        _ham = [item for item in os.listdir(ham_dir) if os.path.isfile(os.path.join(ham_dir, item))]
        number_of_ham = len(_ham)
        number_of_spam = len(_spam)
        paths_spam = [spam_dir+'/'+i for i in _spam]
        paths_ham = [ham_dir+'/'+i for i in _ham]
        self.spam_freq = log_probs(paths_spam)
        self.ham_freq = log_probs(paths_ham)
        self.not_spam_prob = number_of_ham*1.0/(number_of_spam*1.0+number_of_ham*1.0)
        self.spam_prob = number_of_spam*1.0/(number_of_spam*1.0+number_of_ham*1.0)
    
    def is_spam(self, email_path):
        t = load_tokens(email_path)
        test_tokens = t[1]
        test_bigram_tokens = bigram_token(t[1])
        test_tokens.extend(test_bigram_tokens)
        spam_cond_prob = math.log(self.spam_prob)
        not_spam_cond_prob = math.log(self.not_spam_prob)
        
        unknown_ind = 0
        # unigram possibility
        s_s = set(self.spam_freq.keys())
        s_ns = set(self.ham_freq.keys())
        
        #f = open("data.txt",'w')
        for element in test_tokens:
            #f.write(str(spam_cond_prob - not_spam_cond_prob))
            #f.write("\n")
            if element in s_s:
                spam_cond_prob += self.spam_freq[element]
            else:
                if len(element.split()) == 2:
                    spam_cond_prob += self.spam_freq["<UNK>_bi"]
                else:
                    spam_cond_prob += self.spam_freq["<UNK>_uni"]
                        
            if element in s_ns:
                not_spam_cond_prob += self.ham_freq[element]
            else:
                if len(element.split()) == 2:
                    not_spam_cond_prob += self.ham_freq["<UNK>_bi"]
                else:
                    not_spam_cond_prob += self.ham_freq["<UNK>_uni"]

#            # rare-rity of words
#            if element not in s_s:
#                spam_cond_prob += self.spam_freq["illegal_element"]
#            else:
#                spam_cond_prob += self.spam_freq["legal_element"]
#            if element not in s_ns:
#                not_spam_cond_prob += self.ham_freq["illegal_element"]
#            else:
#                not_spam_cond_prob += self.ham_freq["legal_element"]

            if len(element.split()) == 1:
                # capitalization
                if len(element) > 2:
                    if cap_num(element) == 0:
                        spam_cond_prob += self.spam_freq["upper0"]
                        not_spam_cond_prob += self.ham_freq["upper0"]
                    elif cap_num(element) == len(element):
                        spam_cond_prob += self.spam_freq["upper1"]
                        not_spam_cond_prob += self.ham_freq["upper1"]
                    elif element[0].isupper() and cap_num(element) != len(element):
                        spam_cond_prob += self.spam_freq["upper_all"]
                        not_spam_cond_prob += self.ham_freq["upper_all"]
                    else:
                        spam_cond_prob += self.spam_freq["upper_rand"]
                        not_spam_cond_prob += self.ham_freq["upper_rand"]
            
                #length
                if len(element) < 6:
                    spam_cond_prob += self.spam_freq["len_bin_1"]
                    not_spam_cond_prob += self.ham_freq["len_bin_1"]
                elif len(element) >=6 and len(element) < 15:
                    spam_cond_prob += self.spam_freq["len_bin_3"]
                    not_spam_cond_prob += self.ham_freq["len_bin_3"]
                else:
                    spam_cond_prob += self.spam_freq["len_bin_4"]
                    not_spam_cond_prob += self.ham_freq["len_bin_4"]

                # num or not
#                if element.isdigit():
#                    spam_cond_prob += self.spam_freq["digit_only"]
#                    not_spam_cond_prob += self.ham_freq["digit_only"]
#                else:
#                    spam_cond_prob += self.spam_freq["digit_mix"]
#                    not_spam_cond_prob += self.ham_freq["digit_mix"]

#f.close()
        if len(test_tokens) == 1  and test_tokens[0] not in s_s and test_tokens[0] not in s_ns:
            return True
        elif len(test_tokens) < 1:
            return True
        else:
            if spam_cond_prob >= not_spam_cond_prob:
                return True
            else:
                return False



#spam_dir = "data/train/spam"
#ham_dir = "data/train/ham"
#ham_test = "data/dev/ham"
#spam_test = "data/dev/spam"
#_spam = [item for item in os.listdir(spam_dir) if os.path.isfile(os.path.join(spam_dir, item))]
#_ham = [item for item in os.listdir(ham_dir) if os.path.isfile(os.path.join(ham_dir, item))]
#_ham_test = [item for item in os.listdir(ham_test) if os.path.isfile(os.path.join(ham_test, item))]
#_spam_test = [item for item in os.listdir(spam_test) if os.path.isfile(os.path.join(spam_test, item))]
#print "Initializing"
#sf = SpamFilter(spam_dir,ham_dir)
#print "Initialization Finished"
#
#print "***** Now Detecting *****"
#wrong = 0
#for element in _spam_test:
#    if not sf.is_spam("data/dev/spam/"+element):
#        print "Error: The Following Spam Is In Ham Set: "
#        print element
#        wrong += 1
#
#
#for element in _ham_test:
#    if sf.is_spam("data/dev/ham/"+element):
#        print "Error: The Following Ham Is In Spam Set: "
#        print element
#        wrong += 1
#
#print "***** Finish Detecting *****"
#print "Number Of Email That Was Wrong: ", wrong
#print "Correct Percentage: ", ((400.0-wrong)/400)*100,"%"





