# IR-Project
## Organization
The preprocessed files are stored in the [bucket](https://console.cloud.google.com/storage/browser/nir399513) and are pulled into the VM instance at launch.
These preprocessed files include the following, separated into folders:
* body_index/
  - index.pkl
  - .bin files
* title_index/
  - index.pkl
  - .bin files
* anchor_index/
  - index.pkl
  - .bin files
* preprocessed/
  - doctf
  - pages
* pr/
  - csv.gz file
## Project Structure
### read_PL(index, word)
This function was based on assignment two and is used for faster recovery of information.
This was utilized after observing very slow calls to `InvertedIndex.get_postings_iter`.
### search()
The main search utilizes additional stopwords and the BM25 algorithm.
The values which we found led to the best results are as follows:
```
b = 0.75
k1 = 1.2
```
### search_body()
As requested, this function uses TFIDF and cosine similarity.
To do that, we use many functions which are based on assignment 4.
These are the functions in order of access:
```
get_topN_score_for_queries(queries_to_search, index, N=100)
generate_document_tfidf_matrix(query_to_search, index)
get_candidate_documents_and_scores(index, words, pls)
generate_query_tfidf_vector(query_to_search, index)
cosine_similarity(D, Q, query_size)
get_top_n(sim_dict, N=100)
```
### search_title() / search_anchor()
For these functions, a simple utilization of `read_PL` and sorting has given fast results.
### get_pageview()
Access to `pageviews-202108-user.pkl` to extract the needed information.
### get_pagerank()
Filtering through the preprocessed `csv.gz file`
