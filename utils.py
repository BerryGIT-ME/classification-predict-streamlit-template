import string
import pandas as pd
import pickle
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

special_words = pd.read_csv('resources/special_words.csv')['special_words'].to_numpy()
model_names = ["tf_vect.pkl", "logreg2.pkl", "svc_lm2.pkl", "RF2.pkl", "m_log_clf.pkl", "m_svc_lm2.pkl", "m_RF2.pkl", "i_MNB2.pkl"]



# This checks for the occurence of a particular word and returns the count 
# This is helpful as it helps produce the count of the special words we extracted from the word cloud
def count_word_occurrence(word):
    def count(text):
        word_list = text.split(" ")
        word_count = 0
        for item in word_list:
            if item == word:
                word_count = word_count + 1
        return word_count
    return count

# function takes a dataframe and a list of words returns the dataframe with those list of words as new columns
# and the count of those words as the data in the columns
def add_features_from_list(df, feature_list):
    for feature in feature_list:
        df[feature] = df['message'].apply(count_word_occurrence(feature))


#removing web URL
def replace_urls(df, column):
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    subs_url = r'url-web'
    df[column] = df[column].replace(to_replace = pattern_url, value = subs_url, regex = True)
    return 

#converting message column to lower case
def to_lowercase(df, column):
    df['message'] = df['message'].str.lower()
    return

#remove punctuation
def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])

def mbti_lemma(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words.split(" ")]    


def count_words(word):
    word_list = word.split(" ")
    return len(word_list)

def avg_word_length(word):
    string_length =  len(word)
    word_list = word.split(" ")
    word_count = len(word_list)
    return string_length/word_count

def count_citations(word):
    word_list = word.split(" ")
    count = 0 
    
    for word in word_list:
        if word == "urlweb":
            count = count + 1
    return count            

def count_retweets(word):
    word_list = word.split(" ")
    rt_count = 0
    for word in word_list:
        if word == 'rt':
            rt_count = rt_count + 1
    return rt_count

def list_to_string(post):
    return ' '.join(post)

def preprocess(df):
    # checks for special words
    add_features_from_list(df, special_words)
    
    # replace urls
    replace_urls(df, 'message')
    
    # converts to lowercase
    to_lowercase(df, 'message')
    
    #remove punctuation
    df['message'] = df['message'].apply(remove_punctuation)
    
    # tokenize
    tokeniser = TreebankWordTokenizer()
    df['tokens'] = df['message'].apply(tokeniser.tokenize)
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    df['lemma'] = df['message'].apply(mbti_lemma, args=(lemmatizer, ))
    
    # add some metadata
    added_info = df.copy()
    
    # get word count
    added_info['word_count'] = added_info['message'].apply(count_words)
    
    # add avg word length
    added_info['avg_word_length'] = added_info['message'].apply(avg_word_length)
    
    # count citations
    added_info['citations'] = added_info['message'].apply(count_citations)
    
    # count retweets
    added_info['rt_count'] = added_info['message'].apply(count_retweets)
    
    # convert the lemma and tokens value to strings
    added_info['lemma'] = added_info['lemma'].apply(list_to_string)
    added_info['tokens'] = added_info['tokens'].apply(list_to_string)
    
    return added_info

def predict(df):
    df = df.copy()

    with open("resources/tf_vect.pkl", 'rb') as file:
        tf_vect = pickle.load(file)

    X = tf_vect.transform(df.lemma)

    with open("resources/logreg2.pkl", 'rb') as file:
        logreg2 = pickle.load(file)
    with open("resources/svc_lm2.pkl", 'rb') as file:
        svc_lm2 = pickle.load(file)
    with open("resources/RF2.pkl", 'rb') as file:
        RF2 = pickle.load(file)

    sum_predictions = logreg2.predict_proba(X) + svc_lm2.predict_proba(X) + RF2.predict_proba(X)
    
    classes = ['classA', 'classB', 'classC', 'classD']
    
    sum_df = pd.DataFrame(sum_predictions)
    sum_df.columns = classes
    
    meta = df.drop(['message', 'tokens', 'lemma'], axis=1)
    interim = pd.concat([sum_df, meta], axis=1)

    with open("resources/m_log_clf.pkl", 'rb') as file:
        m_log_clf = pickle.load(file)
    with open("resources/m_svc_lm2.pkl", 'rb') as file:
        m_svc_lm2 = pickle.load(file)
    with open("resources/m_RF2.pkl", 'rb') as file:
        m_RF2 = pickle.load(file) 
    with open("resources/i_MNB2.pkl", 'rb') as file:
        i_MNB2 = pickle.load(file)
      
    output = {'Logistic Regression': m_log_clf.predict(interim)[0],
              'Support Vector machine': m_svc_lm2.predict(interim)[0],
              'Random Forest': m_RF2.predict(interim)[0],
              'Naive bayes': i_MNB2.predict(interim)[0],
             }
    return output