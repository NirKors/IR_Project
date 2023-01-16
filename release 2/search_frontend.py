import math
import re
from collections import Counter, defaultdict
import nltk
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from inverted_index_gcp import InvertedIndex
import glob

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

path = "/home/nirkor"

index_body = InvertedIndex.read_index(f"{path}/body_index", "index")
index_body.posting_locs = dict(index_body.posting_locs)
index_title = InvertedIndex.read_index(f"{path}/title_index", "index")
index_title.posting_locs = dict(index_title.posting_locs)
index_anchor = InvertedIndex.read_index(f"{path}/anchor_index", "index")
index_anchor.posting_locs = dict(index_anchor.posting_locs)

path = "/home/nirkor/processed"

with open(f"{path}/processed.pickle", 'rb') as f:
    pages = pickle.load(f)  # id: (title, len(text), count(most_frequent_term))

with open(f"{path}/pageviews-202108-user.pkl", 'rb') as f:
    wiki_id_2_pageview = pickle.loads(f.read())

with open(f"{path}/doctf.pickle", 'rb') as f:
    doctf = pickle.loads(f.read())

with open(f"{path}/pr_results.pickle", 'rb') as f:
    pr_results = pickle.loads(f.read())

pages_len = len(pages)
d_avg = sum([page[1] for page in pages.values()]) / pages_len
info = api.info()
wiki_info = api.info('glove-wiki-gigaword-100')
model = api.load("glove-wiki-gigaword-100")

print("Ready")


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    extended_stopwords = ['ok', 'their', 'before', 'are', 'now', 'until', 's', 'during', 'between', 'not', 'maybe',
                          'an', 'any', 'each', 'can', 'by', 'that', 'from', 'myself', 'than', 'also', 'off', 'these',
                          'they', 'am', 'no', 'will', 'yourself', 'do', 'against', 'out', 'him', 'your', 'whereas',
                          'once', 'have', 'were', 'down', 'its', 'been', 'after', 'could', 'was', 'what', 'doing',
                          'under', 'when', 'only', 'herself', 'always', 'be', 'mine', 'about', 'those', 'ourselves',
                          'our', 'itself', 'then', 'yours', 'in', 'most', 'having', 'we', 'her', 'whose', 'this', 'all',
                          'themselves', 'again', 'his', 'yet', 'further', 'become', 'whoever', 'of', 'neither',
                          'almost', 'else', 'them', 'whether', 't', 'although', 'the', 'why', 'to', 'he', 'yes',
                          'there', 'both', 'so', 'my', 'at', 'had', 'is', 'other', 'below', 'without', 'too',
                          'actually', 'hence', 'it', 'don', 'while', 'wherever', 'she', 'should', 'such', 'above',
                          'and', 'some', 'because', 'but', 'would', 'himself', 'with', 'own', 'became', 'on', 'might',
                          'how', 'few', 'as', 'does', 'may', 'through', 'which', 'very', 'into', 'just', 'a', 'over',
                          'theirs', 'ours', 'whenever', 'nor', 'here', 'did', 'if', 'up', 'must', 'within', 'for', 'me',
                          'has', 'where', 'whom', 'who', 'either', 'yourselves', 'you', 'more', 'or', 'being', 'same',
                          'oh', 'hers', 'i']
    query = request.args.get('query', '')
    query = tokenizer(query)
    query = [token for token in query if token.lower() not in extended_stopwords]
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # We make a document where the key is a tuple with (Doc, Query) key and score value

    query = qexpand(query)

    weight_title = 1
    weight_body = 1 / 2
    weight_anchor = 1 / 4

    query_counter = Counter(query)  # A counter of our queries.
    b = 0.75
    k1 = 1.2  # Usually between [1.2,2]
    N = len(pages)  # Total number of documents in the corpus
    bodies = Counter()
    # Here is how the score function will work:
    for term in query:
        if index_body.df.get(term) and index_body.posting_locs.get(term):
            IDF = np.log10((N + 1) / index_body.df[term])
            TFiq = (k1 + 1) * query_counter[term] / (k1 * query_counter[term])
            pls = read_pls(index_body, term)
            for pl in pls:
                if pl[0] == 0:
                    continue
                B = (1 - b) + (b * pages[pl[0]][1] / d_avg)
                TFij = ((k1 + 1) * pl[1]) / ((B * k1) + pl[1])
                bodies[pl[0]] += IDF * TFij * TFiq

    weighted = Counter()
    _, max_w = bodies.most_common(1)[0]
    for d_id, weight in bodies.most_common(100):
        weighted[d_id] += weight / max_w * weight_body

    # ANCHOR:
    ids = Counter()
    postings = [read_pls(index_anchor, qword) for qword in query]
    for pls in postings:
        if pls:
            for posting in pls:
                ids[posting[0]] = ids.get(posting[0], 0) + 1
    _, max_t_w = ids.most_common(1)[0]
    for d_id, weight in ids.most_common(100):
        weighted[d_id] += weight / max_t_w * weight_anchor
    # ID:
    ids = Counter()
    postings = [read_pls(index_title, qword) for qword in query]
    for pls in postings:
        if pls:
            for posting in pls:
                ids[posting[0]] = ids.get(posting[0], 0) + 1
    _, max_t_w = ids.most_common(1)[0]
    for d_id, weight in ids.most_common(100):
        weighted[d_id] += weight / max_t_w * weight_title

    # PAGE_RANK:

    # PAGE_VIEW:
    # TOTAL:
    for d_id, _ in weighted.most_common(100):
        title = pages[d_id][0]
        res.append((d_id, title))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = tokenizer(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = [word for word in query if word in index_body.posting_locs and index_body.posting_locs[word][0] != 0]
    results = get_topN_score_for_queries(query, index_body)  # Currently (id, cosim)
    for id, _ in results:
        res.append((id, pages[id][0]))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    query = tokenizer(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    ids = {}
    postings = [read_pls(index_title, qword) for qword in query]
    for pls in postings:
        if pls:
            for posting in pls:
                ids[posting[0]] = ids.get(posting[0], 0) + 1

    for id_inner in ids.keys():
        id_in_pages = pages.get(id_inner)
        if id_in_pages:
            res.append(((id_inner, pages[id_inner][0]), ids[id_inner]))
    res = [x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    query = tokenizer(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    ids = {}
    postings = [read_pls(index_anchor, qword) for qword in query]
    for pls in postings:
        if pls:
            for posting in pls:
                ids[posting[0]] = ids.get(posting[0], 0) + 1

    for id_inner in ids.keys():
        id_in_pages = pages.get(id_inner)
        if id_in_pages:
            res.append(((id_inner, pages[id_inner][0]), ids[id_inner]))
    res = [x[0] for x in sorted(res, key=lambda x: x[1], reverse=True)]

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        res.append(pr_results.get(doc_id, 0))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        res.append(wiki_id_2_pageview.get(doc_id, 0))
    # END SOLUTION
    return jsonify(res)


# Staff-provided 3 tokenizer
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenizer(text):
    tokens = np.unique([token.group() for token in RE_WORD.finditer(text.lower()) if token not in all_stopwords])
    return tokens


def get_candidate_documents_and_scores(index, words, pls):
    candidates = {}
    candidates_temp = {}
    i = 0
    for term, pl in zip(words, pls):
        if not pl or pl[0] == 0:
            continue
        normlized_tfidf = [(doc_id, (freq / pages[doc_id][1]) * np.log10(pages_len * index.df[term])) for doc_id, freq
                           in pl if doc_id]

        for doc_id, tfidf in normlized_tfidf:
            if doc_id not in candidates_temp:
                candidates_temp[doc_id] = i
                i += 1
            candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates, candidates_temp


def read_pls(index, word):
    '''
    Based on assignment 2.
    Used for faster recovery of information.
    Args:
        index: InvertedIndex
        word: Word to search

    Returns:
        Posting_List of given word

    '''

    if not index.posting_locs.get(word):
        return
    f_name, offset, n_bytes = index.posting_locs[word][0][0], index.posting_locs[word][0][1], index.df[word] * 6
    with open(index.iname + "/" + f_name, 'rb') as f:
        pls = []
        f.seek(offset)
        for i in range(int(n_bytes / 6)):
            b = (f.read(6))
            doc_id = int.from_bytes(b[0:4], 'big')
            tf = int.from_bytes(b[4:], 'big')
            pls.append((doc_id, tf))
    return pls


def generate_document_tfidf_matrix(query_to_search, index):
    words = []
    pls = []
    for word in query_to_search:
        words.append(word)
        pls.append(read_pls(index, word))

    total_vocab_size = len(words)
    candidates_scores, cand_indices = get_candidate_documents_and_scores(index, words, pls)
    unique_candidates = len(cand_indices)

    D = np.zeros((unique_candidates, total_vocab_size))
    D = pd.DataFrame(D)
    D.index = cand_indices.keys()
    D.columns = words
    query_to_search = list(query_to_search)
    for tup, tfidf in candidates_scores.items():
        doc_id, term = tup
        q_index = query_to_search.index(term)
        D.iloc[cand_indices[doc_id], q_index] = tfidf
    return D


def cosine_similarity(D, Q, query_size):
    D1 = D.dot(Q)
    dict = {}

    for id, cosim_up in D1.items():
        doc_size = doctf[id]
        dict[id] = cosim_up / np.sqrt(query_size * doc_size)
    return dict


def generate_query_tfidf_vector(query_to_search, index):
    C = Counter(query_to_search)
    Qvector = [C[word] / np.log10(index.df[word] * len(query_to_search)) for word in query_to_search]
    return Qvector


def get_top_n(sim_dict, N=100):
    return sorted([(doc_id, np.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                  reverse=True)[:N]


def get_posting_iter(index):
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def get_topN_score_for_queries(queries_to_search, index, N=100):
    D = generate_document_tfidf_matrix(queries_to_search, index)
    Q = generate_query_tfidf_vector(queries_to_search, index)
    sim_dict = cosine_similarity(D, Q, len(queries_to_search))
    ranked = get_top_n(sim_dict, N)
    return ranked


def qexpand(query):
    '''
    Simple query expansion using a prebuilt model.
    Args:
        query: Tokenized query.

    Returns:
        Expanded query.
    '''
    new_q = [x for x in query]
    for qword in query:
        ret = model.most_similar(qword, topn=6)
        for word, _ in ret:
            if word in qword or qword in word:
                new_q.append(word)
    return new_q


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
