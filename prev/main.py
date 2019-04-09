
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
stop_words = set(stopwords.words('english')) 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate,CuDNNLSTM
from tensorflow.keras.layers import Dense, LSTM,Dropout, Activation,Bidirectional,Reshape

'''
Breaking whole data into list of sentences
'''
def seperate_sentences(data):
    return sent_tokenize(data) 


def Sentence_to_POS(all_sentences):

    data=[]
    for sentence in all_sentences: 
        one_sentence_pos = []
        '''
        Word tokenizers is used to find the words and punctuation in a string 
        '''  
        wordsList = nltk.word_tokenize(sentence) 

        '''
        Removing stop words from wordList.
        ''' 
        wordsList = [w for w in wordsList if not w in stop_words]  

        '''
        Using a Tagger. Which is part-of-speech tagger or POS-tagger.  
        '''

        tagged = nltk.pos_tag(wordsList) 
        for x in tagged:
            one_sentence_pos.append(x[1])
        data.append(one_sentence_pos)

    return data


def Fianl_X_and_Y(novel_POS,novel_label,number_of_sentenes_in_one):
    '''
    This funtion takes generates more data for training by breaking one novel into multiple novels
    with k number of sentences (number_of_sentenes_in_one) .we are ignoring some last sentences if they
    dont fit
    '''
    i=0
    X=[]
    Y=[]
    for novel in novel_POS:
        start = 0
        end = number_of_sentenes_in_one
        while(end<=len(novel)):
            X.append(novel[start:end])
            Y.append(novel_label[i])
            start=end
            end=end+number_of_sentenes_in_one
        i+=1
    return X,Y


def Get_X_and_Y_In_POS_Form(path_of_dataset_directory,number_of_sentences_in_one):
    '''
    this function takes the path of directory where the dataset is and returns the processed X and Y.
    The function calls will let u understand more the flow of code
    '''
    all_files = os.listdir(path_of_dataset_directory)
    novel_POS=[]
    novel_label=[]
    for filename in all_files:
        full_path = 'dataset/three_author_dataset/'+filename
        with open(full_path, 'r') as f:
            data = f.read().replace('"\n"','').replace('\n',' ').replace('- ','')
            all_sentences = seperate_sentences(data)
            sentences_to_pos = Sentence_to_POS(all_sentences)
            novel_POS.append(sentences_to_pos)
            novel_label.append(filename[8])
    return Fianl_X_and_Y(novel_POS,novel_label,number_of_sentences_in_one)

def tag_to_index_dictionary(X):
    '''
    This function just creates a dictionary from the POS tags which we
    encountered in our dataset.As the network works with numbers so simple will make
    dictionary which stores the index of associated POS tag
    '''
    tag = set([])
    for doc in X:
        for sentence in doc:
            for word in sentence:
                tag.add(word)
    tag2index = {t: i + 1 for i, t in enumerate(list(tag))}
    tag2index['-PAD-'] = 0
    return tag2index

def convert_tag_to_sequence_numbers(X):
    tag2index = tag_to_index_dictionary(X)
    '''
    using the tag2index dictionary assign indexs to the POS tags in our data
    '''
    new_X=[]
    for doc in X:
        new_S = []
        for sentence in doc:
            new_W=[]
            for word in sentence:
                new_W.append(tag2index[word])
            new_S.append(new_W)
        new_X.append(new_S)
    return new_X


def pad_zeros_to_sequence(X,max_length):
    '''
    This function padds zeros to the sequences so as to make fixed sequences
    According to paper 15 is best value for max_length
    '''
    
    new_X=[]
    for doc in X:
        new_X.append(pad_sequences(doc, maxlen=max_length, padding='post'))
    return new_X

def Encode_Labels(Y):
    '''
    This fucntion encodes the labels i.e., assign numbers to the labels
    '''
    le = LabelEncoder()
    return le.fit_transform(Y)



print("Preprocessing Phase 1 : Fetching data")
print(".....................................")

X,Y = Get_X_and_Y_In_POS_Form('dataset/three_author_dataset',100)

print("Preprocessing Phase 1 : Finished\n\n")
print(".....................................")
print("\nPreprocessing Phase 1 : Processing data")
print(".....................................")

A,B,C=0,0,0
X_data=[]
for i in range(len(X)):
    if(Y[i]=='A' and A<95):
        X_data.append([X[i],Y[i]])
        A+=1
    elif(Y[i]=='B' and B<95):
        X_data.append([X[i],Y[i]])
        B+=1
    elif(Y[i]=='C' and C<95):
        X_data.append([X[i],Y[i]])
        C+=1
import random
random.shuffle(X_data)
new_X=[]
new_Y=[]
for m , n in X_data:
    new_X.append(m)
    new_Y.append(n)

X=new_X
Y=new_Y
DICTIONARY_LENGTH = len(tag_to_index_dictionary(X))
MAX_LENGTH = 30
SENTENCES_IN_NOVEL=100

X = convert_tag_to_sequence_numbers(X)
X = pad_zeros_to_sequence(X,MAX_LENGTH)
X = np.array(X)
TOTAL_ROWS=X.shape[0]

real_X = []
for i in range(SENTENCES_IN_NOVEL):
    real_X.append(list())
for i in range(SENTENCES_IN_NOVEL):
    for j in range(TOTAL_ROWS):
        real_X[i].append(X[j][i])

X_train = []
for i in range(SENTENCES_IN_NOVEL):
    X_train.append(np.array(real_X[i]))



Y = Encode_Labels(Y) 
from tensorflow.keras.utils import to_categorical
y_binary = to_categorical(Y)



print("Preprocessing Phase 2 : Finished\n\n")
print(".....................................")
print("Shape of Actual X ",np.array(real_X).shape)
print("Shape of Changed X ",np.array(real_X).shape)
print("Shape of Y" ,y_binary.shape)
print(".....................................")
print("\n\nMode Construction")
print(".....................")

outputs=[]
inputs_=[]

for i in range(SENTENCES_IN_NOVEL):
    inputlayer = Input(shape=[MAX_LENGTH])
    inputs_.append(inputlayer)
    layer = Embedding(DICTIONARY_LENGTH, 50, input_length=30)(inputlayer)
    layer = LSTM(50)(layer)
    layer = Dense(50)(layer)
    layer = Activation('relu')(layer)
    outputs.append(layer)
merge = concatenate(outputs)
merge = Reshape((SENTENCES_IN_NOVEL, 50), input_shape=(SENTENCES_IN_NOVEL*50,))(merge)
merge = Bidirectional(LSTM(50))(merge)
hidden1 = Dense(10, activation='relu')(merge)
out = Dense(3, activation='softmax')(hidden1)
model = Model(inputs=inputs_,outputs=out)
model.summary()