
# coding: utf-8

# In[8]:


import logging, math, pickle
import numpy as np
from gensim.models import word2vec
import jieba
from random import randint
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request
app = Flask(__name__)


# In[9]:


def gen_avg_data(model):
    logging.info('start genering sentence average...')
    tfidf_matrix, word_bag = gen_tfidf_matrix()
    arr = []
    with open("question_seg_stopwords.txt", 'r', encoding='utf8') as file:
        for idx, line in enumerate(file.readlines()):
#             logging.info('line=%s', str(idx))
            words = line.strip().split(" ")
            avg = count_avg_tfidf(idx, words, model, tfidf_matrix, word_bag)
            arr.append(avg)
    all_avg = np.array(arr)
    np.save("sent_avg_tfidf_200.npy", all_avg)
    logging.info('end genering sentence average...')

def sent2vec_simple(sent, model):
    words = jieba.cut(sent, cut_all=False)
    return count_avg_simple(words, model)
    

def count_avg_simple(words, model):
    emb_cnt = 0
    avg_dlg_emb = np.zeros((200,))
    for word in words:
        if word in model.wv.vocab:
            avg_dlg_emb += model[word]
            emb_cnt += 1
    if emb_cnt != 0:
        avg_dlg_emb /= emb_cnt
    return avg_dlg_emb 
    
def sent2vec(sent, model, tfidf_matrix, word_bag_all):
    words = jieba.cut(sent, cut_all=False)
    words = list(words) # words是generator只能for一次
    return count_avg_newsent(words, model, tfidf_matrix, word_bag_all)
    

def count_avg_newsent(words, model, tfidf_matrix, word_bag_all):
    emb_cnt = 0
    avg_dlg_emb = np.zeros((200,))
    
    # 取得輸入句子的frequency *& word_bag
    vectorizer = CountVectorizer()  
    frequency = vectorizer.fit_transform([concat_words(words)])  
    word_bag_sent = vectorizer.get_feature_names() 
    
    for word in words:
        if word in model.wv.vocab:
            emb_cnt += 1
            tf = count_tf(word, frequency, word_bag_sent)
            avg_dlg_emb += model[word]*count_tfidf(word, tf, tfidf_matrix, word_bag_all)

    if emb_cnt != 0:
        avg_dlg_emb /= emb_cnt
    return avg_dlg_emb

def count_avg_tfidf(idx, words, model, tfidf_matrix, word_bag):
    emb_cnt = 0
    avg_dlg_emb = np.zeros((200,))
    for word in words:
        if word in model.wv.vocab:
            emb_cnt += 1
#     tfidf = get_tfidf(tfidf_matrix, word_bag, idx, word)
#     if tfidf != 0:
#         avg_dlg_emb += model[word]*tfidf
    
    arr1, arr2 = tfidf_matrix[idx].nonzero()
    for i in arr2:
        w = word_bag[i]
        if w in model.wv.vocab:
            avg_dlg_emb += model[w]*tfidf_matrix[idx][0,i]
    
    if emb_cnt != 0:
        avg_dlg_emb /= emb_cnt
    return avg_dlg_emb

def gen_tfidf_matrix():
    logging.info('start generate tfidf matrix')
    corpus = []
    with open("question_seg_stopwords.txt", 'r', encoding='utf8') as file:
        for line in file.readlines():
            corpus.append(line)
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf_matrix=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word_bag=vectorizer.get_feature_names()
    with open('tfidf_matrix.pkl', 'wb') as output:
        logging.info('saving tfidf_matrix object...')
        pickle.dump(tfidf_matrix, output, pickle.HIGHEST_PROTOCOL)
    with open('word_bag.pkl', 'wb') as output:
        logging.info('saving word_bag object...')
        pickle.dump(word_bag, output, pickle.HIGHEST_PROTOCOL)
    logging.info('end generate tfidf matrix')
    return tfidf_matrix, word_bag
    
def get_tfidf(tfidf_matrix, word_bag, line, word):
    try:
        index = word_bag.index(word)
        return tfidf_matrix[line][0, index]
    except:
        return 0
    
def concat_words(words):
    sent = ""
    for w in words:
        sent = sent + w +" "
    return sent.strip()

def count_tf(word, frequency, word_bag):
    frequency = frequency.toarray()
    if word in word_bag:
        return frequency[0][word_bag.index(word)]/sum(frequency[0])
    else: 
        return 0


# tf = 新句子的字的tf
# tfidf_matrix = 不包含新句子的matrix
def count_tfidf(word, tf, tfidf_matrix, word_bag):
    sent_count = tfidf_matrix.shape[0]+1
    if word not in word_bag:
        return 0
    word_bag_idx = word_bag.index(word)
    arr1, arr2 = tfidf_matrix[:, word_bag_idx].nonzero()
    sent_include_word_count = len(arr1)+1
    idf = math.log(sent_count/sent_include_word_count+1)
    return tf*idf

def most_similar_answer(question, ans_list, model):
    q_vec = sent2vec_simple(question, model)
    a_vecs = []
    for ans in ans_list:
        a_vec = sent2vec_simple(ans, model)
        a_vecs.append(a_vec)
    a_vecs = np.array(a_vecs)    
    similarity = a_vecs.dot(q_vec)/ np.linalg.norm(a_vecs, axis=1) / np.linalg.norm(q_vec)
    max_idx = find_max_idx(similarity)
    return ans_list[max_idx]

def find_max_idx(similarity): # 遇到nan的情況無法使用argmax
    max_idx = 0
    max_s = 0
    for i, s in enumerate(similarity):
        if s > max_s:
            max_s = s
            max_idx = i
    return max_idx

def get_answer(question):
    q_avg = sent2vec(question, model, tfidf_matrix, word_bag)
    similarity = all_avg.dot(q_avg)/ np.linalg.norm(all_avg, axis=1) / np.linalg.norm(q_avg)
    max_idx = find_max_idx(similarity)
    ans = answer_lines[max_idx].split("@@##@@##@@!")
    return most_similar_answer(input_text, ans, model)

@app.route('/chatbot/', methods=['GET'])
def find_answer():
    question = request.args.get('question')
    logging.info('receive question = %s', question)
    answer = get_answer(question)
    logging.info('return answer = %s', answer)
    return answer


# In[10]:


if __name__ == '__main__':
    global tfidf_matrix
    global word_bag
    global all_avg
    global answer_lines
    global model
#     jieba.set_dictionary('jieba/dict.txt')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = word2vec.Word2Vec.load("word2vec_stopword_all_200_10_sg0.model")
#     gen_avg_data(model)

    
    with open('tfidf_matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)
    with open('word_bag.pkl', 'rb') as file:
        word_bag = pickle.load(file)     
        
    all_avg = np.load("sent_avg_tfidf_200.npy")
    input_text = ""
    question_file = open("question_seg.txt", 'r', encoding='utf8')
    answer_file = open("answers.txt", 'r', encoding='utf8')
    question_lines = question_file.readlines()
    answer_lines = answer_file.readlines()
    
    app.run(debug=True)
#     input_text = input()
#     print(get_answer(input_text))
#     input_text = input()
#     while input_text != "exit":
#         q_avg = sent2vec(input_text, model, tfidf_matrix, word_bag)
#         similarity = all_avg.dot(q_avg)/ np.linalg.norm(all_avg, axis=1) / np.linalg.norm(q_avg)
#         # 用argmax遇到nan會GG
#         max_idx = find_max_idx(similarity)
# #         max_idx = similarity.argmax()        
#         print("最相近的題目：", max_idx, question_lines[max_idx])
#         ans = answer_lines[max_idx].split("@@##@@##@@!")
# #         rand = randint(0,len(ans)-1)
# #         print("Ans：", ans[rand])
# #         print("")
#         print(max_idx, most_similar_answer(input_text, ans, model))
#         input_text = input()


# In[33]:


# lis = np.argsort(similarity)[-10:]
# # for i in np.flipud(similarity):
# #     print(i)
# #     break
# for num in lis :
#     print (num, question_lines[num])


# In[31]:


# tfidf_matrix =pickle.load(file)
# word_bag =pickle.load(file)     
# print(word_bag.index("墾丁"))
# print(word_bag[117400])
# print(tfidf_matrix[29259])

