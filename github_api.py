
# coding: utf-8

# In[1]:


from github import Github
import pandas as pd
import networkx as nx
#import matplotlib as plt
from IPython.display import display_svg
import csv
import os
import numpy as np
import re
import bs4
import sys
import MeCab
#import urllib.request
from pprint import pprint
from gensim.models import word2vec
import networkx as nw
import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain


# In[2]:


# アカウントでログイン
#gh = Github('tnmr', '************')

# トークンでログイン
# tnmrのアカウントのトークン
token = "ae43b8e573ffc42495adccfc410241fb40ab2506"
# Study-Communicationアカウントのトークン
#token = "3967d65e5e98e587b4f85eb73266ef103527292d"
gh = Github(token)

repo_num = 0
# 検索ワードのスターの少ない順に表示（検索ワード、ソート、昇順降順）
for repo in gh.search_repositories("vim-jp", "stars", "asc"):
    print(repo.full_name)
    repo_num = repo_num + 1

print(repo_num)


# In[3]:


# スターが最も多いリポジトリのissueのナンバーとタイトルを表示
issue_num = 0
for issue in repo.get_issues():
    print('#{:<5} {}'.format(issue.number, issue.title))
    issue_num = issue_num + 1
    
print(issue_num)


# In[4]:


# 指定したissueのタイトルと@ユーザ名とコメントを表示
var = 12   # ←issue番号指定
comment_num = 0
issue = repo.get_issues()[var]
print('@' + issue.user.login)
print('#{:<5} {}'.format(issue.number, issue.title))
#print(str(issue.body).replace('\n', ''))
print(issue.body)
for comment in issue.get_comments():
    print('@' + comment.user.login)
    #print(str(comment.body).replace('\n', ''))
    print(comment.body)
    comment_num = comment_num + 1

print(comment_num)


# In[5]:


# issue番号を指定してissue表示
var = 12
issue = repo.get_issues()[var]
users = []
users.append(issue.user.login)
comments = []
# 改行削除
comments.append(str(issue.title) + '。' + str(issue.body).replace('\n', ''))
for comment in issue.get_comments():
    users.append(comment.user.login)
    #comments.append(comment.body)
    # 改行削除ver
    comments.append(str(comment.body).replace('\n', ''))

print(users)
print(comments)


# In[6]:


# 引用部分確認
for comment in issue.get_comments():
    quote = re.findall("(?<=\> ).*?(?=\\\n)", comment.body)
    print(quote)    


# In[7]:


# メンション部分確認
for comment in issue.get_comments():
    mention = re.findall("(?<=^\@).*?(?= )", comment.body)
    print(mention)


# In[8]:


# 引用部分のエッジ追加
quote_edge = []
for j, comment in enumerate(issue.get_comments()):
    quotes = re.findall("(?<=\> ).*?(?=\\\n)", comment.body)
    if quotes:
        source = []
        for quote in quotes:
            for i, comment_source in enumerate(issue.get_comments()):
                if quote in comment_source.body:
                    if i != j:
                        source.append([i, j])
                    break
                    
        source = list(map(list, set(map(tuple, source))))
        source = list(chain.from_iterable(source))
        if source:
            quote_edge.append(source)

print(quote_edge)


# In[9]:


# メンション部分のエッジ追加
mention_edge = []
for j, comment in enumerate(issue.get_comments()):
    mention = re.findall("(?<=^\@).*?(?= )", comment.body)
    if mention:
        source = []
        i = j
        while i > -1:
            if users[i] in mention:
                mention_edge.append([i-1, j])
                break
            i = i -1
        
        if source:
            mention_edge.append(source)
            
print(mention_edge)


# In[12]:


# エッジ作成
defined_edge = quote_edge + mention_edge
defined_edge = sorted(list(map(list, set(map(tuple, defined_edge)))), key=lambda x: x[1])
print("引用+メンション")
print(defined_edge)

edge = []
for i, comment in enumerate(issue.get_comments()):
    edge.append([i, i+1])
edge.pop()
print("i, i+1")
print(edge)

# 引用、メンションのエッジは置き換える
for i in range(len(defined_edge)):
    edge[defined_edge[i][1]-1] = defined_edge[i]
print("更新されたエッジ")
print(edge)


# In[94]:


# TFIDF

# 指定したカテゴリの単語のみを取得
def get_words(text):
    # 辞書の指定
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    words = []
    # ターゲットの指定
    target_categories = ["名詞", "動詞",  "形容詞", "副詞", "連体詞", "感動詞"]
    # 省きたい単語
    #target_words = ["ー"]
    # 分かち書き
    node = m.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        # "ー"を単語から省く
        #if not node.surface in target_words:
            # 指定したカテゴリの単語をリストに追加
        if fields[0] in target_categories:
            words.append(node.surface)
        node = node.next

    words_str = map(str, words)
    return ",".join(words_str)

# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# In[8]:


# 単語抽出
doc = []
for i in range(len(comments)):
    doc.append(get_words(comments[i]))

docs = np.array(doc)

# TFIDFで文章をベクトル化
np.set_printoptions(precision=3)
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)
 
for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
    print(k,v)
    
print(vecs.toarray())


# In[10]:


# TFIDFでベクトル化された文章の類似度を計算

# 類似度の閾値設定
threshold = 0.2

for x in range(len(docs)):
    for y in range(len(docs)):
        if x != y and x < y:
            if not comments[x] == comments[y]:
                if cos_sim(vecs.toarray()[x], vecs.toarray()[y]) > threshold:
                    print(str(x) +  "：" + comments[x] + "=>" + str(y) + "：" + comments[y])
                    print(cos_sim(vecs.toarray()[x], vecs.toarray()[y]))


# In[33]:


g = nx.DiGraph()
for node in range(len(comments)):
    g.add_node(comments[node][:5])
    
#plt.figure(figsize=(15, 15))
#pos = nx.spring_layout(g)
#nx.draw_networkx(g, pos)

svg = nx.nx_agraph.to_agraph(g).draw(prog='fdp', format='svg')
display_svg(svg, raw=True)


# In[12]:


#辞書の指定
m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
#m = MeCab.Tagger("-Ochasen")

#print('@'+issue.user.login)
#print(m.parse(issue.body))
#for comment in issue.get_comments():
#    print('@'+comment.user.login)
#    print(m.parse(comment.body))

for i in range(len(users)):
    print(users[i])
    print(m.parse(comments[i]))


# In[21]:


# Word2Vecライブラリのロード

# size: 圧縮次元数
# min_count: 出現頻度の低いものをカットする
# window: 前後の単語を拾う際の窓の広さを決める
# iter: 機械学習の繰り返し回数(デフォルト:5)十分学習できていないときにこの値を調整する
# model.wv.most_similarの結果が1に近いものばかりで、model.dict['wv']のベクトル値が小さい値ばかりのときは、学習回数が少ないと考えられます。
# その場合、iterの値を大きくして、再度学習を行います。

# 事前準備したword_listを使ってWord2Vecの学習実施
# 単語リストは少ない量でないとダメ？
model = word2vec.Word2Vec(, size=100,min_count=5,window=5,iter=100)
model.save("word2vec.moedl")


# In[1]:


import matplotlib
print(matplotlib.rcParams['font.family'])

