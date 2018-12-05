
# coding: utf-8

# In[16]:


# インポート
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab


# In[17]:


# 発言情報ファイルの読み込み
file_num = 4   # ←この数値を変える
file_info = ("study_data/com_info%s.csv" % file_num)
com_info = pd.read_csv(file_info, names=('発言者', '発言時刻', 'UNIX TIME', '発言内容'), index_col=0, sep=',', encoding='shift-jis')
com_info


# In[67]:


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


# In[69]:


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

#print(docs)
#print(len(docs))

pd.DataFrame(docs)


# In[70]:


# TFIDFで文章をベクトル化
np.set_printoptions(precision=2)
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)
 
for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
    print(k,v)
    
print(vecs.toarray())
#print(vecs)


# In[82]:


# TFIDFでベクトル化された文章の類似度を計算

# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

threshold = 0.15

#sim = np.zeros((len(docs), len(docs)))
for x in range(len(docs)):
    for y in range(len(docs)):
        if x != y and x < y:
            if not com_info.iat[text_num[x], 3] == com_info.iat[text_num[y], 3]:
                if cos_sim(vecs.toarray()[x], vecs.toarray()[y]) > threshold:
                    print(str(text_num[x]) +  "：" + com_info.iat[text_num[x], 3] + "=>" + str(text_num[y]) + "：" + com_info.iat[text_num[y], 3] )
                    print(cos_sim(vecs.toarray()[x], vecs.toarray()[y]))
                #sim[x, y] = cos_sim(vecs.toarray()[x], vecs.toarray()[y])

#print(len(sim))

#df = pd.DataFrame(sim)
#df.to_csv("similarity.csv", header=False, index=False)

