
# coding: utf-8

# In[9]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import MeCab
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer


# In[10]:


# ファイル読み込み
file_num = 4   # ←この数値を変える
file_net = ("study_data/adjacent_matrix%s.csv" % file_num)
com_net = pd.read_csv(file_net, header=0, index_col=0, sep=',')
com_net


# In[11]:


# 発言情報ファイルの読み込み
file_info = ("study_data/com_info%s.csv" % file_num)
com_info = pd.read_csv(file_info, names=('発言者', '発言時刻', 'UNIX TIME', '発言内容'), index_col=0, sep=',', encoding='shift-jis')
com_info


# In[12]:


# エッジリストの作成
edges = []
for x in range(len(com_net)):
    for y in range(len(com_net.columns)):
        if(com_net.iloc[x, y] == 1):
            edges.append([x, y])
            
print(edges)


# In[13]:


# Word2Vec 文の類似度算出

# neologdを使ってモデル作成
m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
m.parse('')
model = word2vec.Word2Vec.load("wikiextractor-master/wiki.model")

# 名詞、動詞、形容詞に限定
target_categories = ["名詞", "動詞",  "形容詞"]

# テキストのベクトルを計算
def get_vector(text):
    sum_vec = np.zeros(200)
    word_count = 0
    node = m.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        if fields[0] in target_categories:
            sum_vec += model.wv[node.surface]
            word_count += 1
        node = node.next

    return sum_vec / word_count

# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# In[14]:


# Word2Vec 発話と応答の類似度の抽出（定義した発話→応答文のみ）
for i in edges:
        enc, dec = i
        try:
            get_vector(com_info.iat[enc, 3])
            get_vector(com_info.iat[dec, 3])
        except Exception as e:
            print(com_info.iat[enc, 3] + "=>" + com_info.iat[dec, 3])
            print(e)
        else:
            print(com_info.iat[enc, 3] + "=>" + com_info.iat[dec, 3])
            v1 = get_vector(com_info.iat[enc, 3])
            v2 = get_vector(com_info.iat[dec, 3])
            print(cos_sim(v1, v2))


# In[15]:


# Word2Vec 発話と応答の類似度の抽出（全発話総当たり）
for x in range(len(com_net)):
    for y in range(len(com_net.columns)):
        if x!=y and x < y:
            try:
                get_vector(com_info.iat[x, 3])
                get_vector(com_info.iat[y, 3])
            except Exception as e:
                print(com_info.iat[x, 3] + "=>" + com_info.iat[y, 3])
                print(e)
            else:
                print(com_info.iat[x, 3] + "=>" + com_info.iat[y, 3])
                v1 = get_vector(com_info.iat[x, 3])
                v2 = get_vector(com_info.iat[y, 3])
                print(cos_sim(v1, v2))


# In[16]:


# TFIDF算出

# 指定したカテゴリの単語のみを取得
def get_words(text):
    # 辞書の指定
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    words = []
    # ターゲットの指定
    target_categories = ["名詞", "動詞",  "形容詞", "副詞", "連体詞", "感動詞"]
    # 省きたい単語
    target_words = ["ー"]
    # 分かち書き
    node = m.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        # "ー"を単語から省く
        if not node.surface in target_words:
            # 指定したカテゴリの単語をリストに追加
            if fields[0] in target_categories:
                words.append(node.surface)
        node = node.next

    words_str = map(str, words)
    return ",".join(words_str)


# In[17]:


# 配列に変換
text_num = []
doc = []
head = ["（スタンプ）", "（画像）", "（ノート）", "（URL）", "（ファイル）"]

for i in range(len(com_info)):
    # （）で始まる文章の削除
    flag = True
    for j in head:
        if j in com_info.iat[i, 3]:
            flag = False
    
    if flag:
        text_num.append(i)
        doc.append(get_words(com_info.iat[i, 3]))

        
docs = np.array(doc)

# TFIDFで文章をベクトル化
np.set_printoptions(precision=2)
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)
 
for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
    print(k,v)
    
print(vecs.toarray())


# In[18]:


# TFIDFでベクトル化された文章の類似度を計算

# 類似度の閾値設定
threshold = 0.15

for x in range(len(docs)):
    for y in range(len(docs)):
        if x != y and x < y:
            if not com_info.iat[text_num[x], 3] == com_info.iat[text_num[y], 3]:
                if cos_sim(vecs.toarray()[x], vecs.toarray()[y]) > threshold:
                    print(str(text_num[x]) +  "：" + com_info.iat[text_num[x], 3] + "=>" + str(text_num[y]) + "：" + com_info.iat[text_num[y], 3] )
                    print(cos_sim(vecs.toarray()[x], vecs.toarray()[y]))


# In[19]:


# グラフ作成、可視化
g = nx.DiGraph()
g.add_nodes_from(com_net)
g.add_edges_from(edges)

nx.draw_networkx(g, pos=nx.spring_layout(g), node_size=10, width=0.1)
#plt.figure(figsize=(10,10), dpi=200)
plt.savefig('network.png')
plt.show()

