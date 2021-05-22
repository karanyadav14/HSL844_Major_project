#!/usr/bin/env python
# coding: utf-8

# ### Candidates and Queries
# 

# In[5]:


import codecs

def normalize_term(term):
    """ Normalize a query or candidate: lower-case and replace spaces with underscores."""
    
    return term.lower().replace(" ", "_")

def load_queries(file, normalize=True):
    """Given query file, return list of queries and list of query types.
    
    """
    f=open(file)                                  #Opening given file
    queries = []   
    query_types = []
    for line in f:                                    
        q, qtype = line.strip().split("\t")      #Seperating the given query word and its type from file
        if normalize:                            #Checking if query word is normalized
            q = normalize_term(q)                #Normalize the query word 
        queries.append(q)                        #Appending query word in a list 
        query_types.append(qtype)                #Appending query type in another list
    return queries, query_types



def load_candidates(file, normalize=True):
    """Given a file, return list of candidates.
    
    """
    with codecs.open(file, "r", encoding="utf-8") as f:      #Opening file with unicode strings with codecs
        candidates = []
        for line in f:                                    
            a = line.strip()                                
            if len(a):
                if normalize:
                    a = normalize_term(a)
                candidates.append(a)                        #Appending candidate hypernym to list
        return candidates



        
def load_vocab(file, lower_queries=False):
    """ Given a data file load candidates and queries (training, trial, and test). 
    Return set of candidates and set of queries.
    """
    candidates = set(load_candidates(file, normalize=False))
    
    list1=[]                                                   #Collecting all query words in single list
    f1=load_queries("1A.english.test.data.txt")  
    f2=load_queries("1A.english.training.data.txt")
    f3=load_queries("1A.english.trial.data.txt")

    for i in [f1,f2,f3]:
        list1.extend(i[0])
    
    queries=set(list1)
    return queries, candidates




print("Candidates and Queries are:")
queries, candidates = load_vocab('1A.english.vocabulary.txt') 
print("No of candidates: {}".format(len(candidates)))
print("No of queries: {}".format(len(queries)))

vocab = candidates.union(queries)                                #Collecting candidates and queries in single list
print("Size of vocab: {}".format(len(vocab)))
trigrams = set()
bigrams = set()
unigrams = set()
for i in vocab:
    Num_words = len(i.split())
    if Num_words == 3:
        trigrams.add(i)                                        #Creating trigram list from vocab
    elif Num_words == 2:
        bigrams.add(i)                                         #Creating bigram list from vocab
    elif Num_words == 1:
        unigrams.add(i)
    else:
        msg = "Error: '{}' is not unigram, bigram or trigram".format(term)
        raise ValueError(msg)                                  #Raising error if no n-grams found
print("No of unigrams: {}".format(len(unigrams)))
print("No of bigrams: {}".format(len(bigrams)))
print("No of trigrams: {}".format(len(trigrams)))


# ### Corpus Processing

# In[6]:


import codecs
from collections import defaultdict

print("Counting lines in corpus...")
f10=codecs.open('Short_corpus.txt',"r", encoding="utf-8")      #Reading file using codecs
Num_lines = sum(1 for line in f10)                             #Counting number of lines in file

print("Counting n-gram frequencies in corpus...")
term_to_freq_in = defaultdict(int)                             
line_count = 0
with codecs.open('Short_corpus.txt', "r", encoding="utf-8") as f_1:
    for line in f_1:
        line_count += 1                                       #Counting lines in file
        if line_count % 100000 == 0:                          #Checking for every million lines
            msg = "{}/{} lines processed.".format(line_count, Num_lines)          
            msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_in), len(vocab))
            print(msg)                                         #Printing processed lines
        line = line.strip().replace("_", "")         
        words = [w.lower() for w in line.split()]               #Adding lines into list
        for n in [1,2,3]:
            for i in range(len(words)+n-1):
                term = " ".join(words[i:i+n])
                if term in vocab:
                    term_to_freq_in[term] += 1                 #Calculating frequency of each word
msg = "{}/{} lines processed.".format(line_count, Num_lines)
msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_in), len(vocab))
print(msg)
No_missing_q = sum(1 for w in queries if term_to_freq_in[w] == 0)       #Print if querries are not found in corpus
No_missing_c = sum(1 for w in candidates if term_to_freq_in[w] == 0)    #Print if candidates are not found in corpus
print("Nb zero-frequency queries: {}".format(No_missing_q))
print("Nb zero-frequency candidates: {}".format(No_missing_c))





# In[7]:



def extract_ngrams(tokens, n, ngram_vocab, term_to_freq):
    """ Given a list of tokens and a vocab of n-grams, extract list of
    non-overlapping n-grams found in tokens.
    """
    ngrams_found = []                                  
    for i in range(len(tokens)-n+1):                 #Looking in range of tokens
        term = " ".join(tokens[i:i+n])               
        if term in ngram_vocab:                     
            ngrams_found.append((i,term))
    if len(ngrams_found) < 2:
        return ngrams_found
    
    ngrams_filtered = ngrams_found[:1]               # Eliminating overlap
    for (start, term) in ngrams_found[1:]:           #Looking for n-grams in start and end term list
        prev_start, prev_term = ngrams_filtered[-1]
        if start - prev_start < n:
            if term not in term_to_freq or term_to_freq[term] < term_to_freq[prev_term]:
                ngrams_filtered[-1] = (start, term)
        else:
            ngrams_filtered.append((start, term))
    return ngrams_filtered

def get_formatted_sample(strings, max_sampled):
    sub = strings[:max_sampled]
    if len(strings) > max_sampled:
        sub.append("... ({}) more".format(len(strings)-max_sampled))
    return ", ".join(sub)

def get_indices_unmasked_spans(mask):
    """ Given a mask array, return spans of unmasked list items."""
    spans = []
    start = 0
    while start < len(mask):
        if mask[start]:
            start += 1
            continue
        end = start
        for i in range(start+1, len(mask)):
            if mask[i]:
                break
            else:
                end = i
        spans.append((start, end))
        start = end + 1
    return spans


# In[9]:


#print("Counting lines in corpus...")
f10=codecs.open('Short_corpus.txt',"r", encoding="utf-8")       #Open corpus file   
Num_lines = sum(1 for line in f10)                              #Counting number of lines

from collections import defaultdict

# Replace multi-word terms with single tokens
print("\nProcessing corpus...")
term_to_freq_out = defaultdict(int)
line_count = 0
with codecs.open('Short_corpus.txt', "r", encoding="utf-8") as f_in, codecs.open('output_file.txt', "w", encoding="utf-8") as f_out:
    for line in f_in:                                         #For lines in corpus
        line_count += 1                                       #Counting the line in corpus
        if line_count % 100000 == 0:                          #Keeping track of line in corpus
            msg = "{}/{} lines processed.".format(line_count, Num_lines)
            msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_out), len(vocab))
            print(msg)
        line = line.strip().replace("_", "")                 #Separating lines with comma
        words = [w.lower() for w in line.split()]            #Adding lines into list words    
        
        
        term_lengths = [0 for _ in range(len(words))] #Making list indicating the length of the term found at each position    
        masked_indices = [0 for _ in range(len(words))]
        
        # Check for trigrams
        trigrams_found = extract_ngrams(words, 3, trigrams, term_to_freq_in) #Extracting trigrams from list words
        for (i, term) in trigrams_found:                                     
            term_lengths[i] = 3                             
            term_to_freq_out[term] += 1
            masked_indices[i] = 1
            masked_indices[i+1] = 1
            masked_indices[i+2] = 1
        
        # Check for bigrams
        for (beg, end) in get_indices_unmasked_spans(masked_indices):    
            bigrams_found = extract_ngrams(words[beg:end+1], 2, bigrams, term_to_freq_in)                
            for (i, term) in bigrams_found:
                term_lengths[beg+i] = 2
                term_to_freq_out[term] += 1
                masked_indices[beg+i] = 1
                masked_indices[beg+i+1] = 1
        
        # Check for unigrams
        for (beg, end) in get_indices_unmasked_spans(masked_indices):    
            for i in range(beg,end+1):
                term = words[i]
                if term in unigrams:
                    term_to_freq_out[term] += 1
                    term_lengths[i] = 1
        
        
        norm_terms = []                             # Writing sentence in output file
        i = 0
        while i < len(term_lengths):
            n = term_lengths[i]                    #Checking the length of term in list term_length
            if n > 1:
                norm_terms.append("_".join(words[i:i+n]))  
                i += n
            else:
                if n == 0:                        #Checking if n is zero or not
                    norm_term = "<UNK>"           #If zero add unknown to list
                else:
                    norm_term = words[i]
                norm_terms.append(norm_term)
                i += 1
        sent = " ".join(norm_terms)             #Joining norm terms
        f_out.write(sent+"\n")                  #Writing in output file
        
msg = "{}/{} lines processed.".format(line_count, Num_lines)             #Keeping track of processed lines
msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_out), len(vocab))
print(msg)

missing_q = [w for w in queries if term_to_freq_out[w] == 0]       #Checking for missed queries
missing_c = [w for w in candidates if term_to_freq_out[w] == 0]    #Checking for missed candidates
print("Nb missing queries in output: {}".format(len(missing_q)))
max_shown = 200
if len(missing_q):
    msg = "Examples: {}".format(get_formatted_sample(sorted(missing_q), max_shown))  #Print examples of missing queries
    print(msg)
print("Nb missing candidates in output: {}".format(len(missing_c)))                #Print examples of missing candidate
if len(missing_c):
    msg = "Examples: {}".format(get_formatted_sample(sorted(missing_c), max_shown))
    print(msg)


# Write frequencies
with open('output_file.txt', "w", encoding="utf-8") as f:              #Writing in output processed file
    for term, freq in sorted(term_to_freq_out.items(), key=lambda x:x[0]):
        # Normalize term
        term_norm = "_".join(term.split())
        f.write("{}\t{}\n".format(term_norm, freq))
msg = "Wrote vocab --> {}".format('output_file.txt')
print(msg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




