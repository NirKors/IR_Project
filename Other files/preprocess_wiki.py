import numpy as np
import pandas as pd
%load_ext google.colab.data_table
import bz2
from functools import partial
from collections import Counter, OrderedDict
import pickle
import heapq
from itertools import islice, count, groupby
from xml.etree import ElementTree
import codecs
import csv
import os
import re
import gzip
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
%matplotlib inline
from pathlib import Path
import itertools
from time import time
import hashlib

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

try:
    import mwparserfromhell as mwp
except ImportError:
    !pip install -I --compile mwparserfromhell==0.6.0
finally:
    import mwparserfromhell as mwp
mwp.definitions.INVISIBLE_TAGS.append('ref')

## Download one wikipedia file
part_url = 'https://dumps.wikimedia.org/enwiki/20210801/enwiki-20210801-pages-articles-multistream15.xml-p17324603p17460152.bz2'
wiki_file = Path(part_url).name
!wget -N $part_url

def page_iter(wiki_file):
  """ Reads a wiki dump file and create a generator that yields one page at a 
      time. 
  Parameters:
  -----------
  wiki_file: str
    A path to wiki dump file.
  Returns:
  --------
  tuple
    containing three elements: article id, title, and body. 
  """
  # open compressed bz2 dump file
  with bz2.open(wiki_file, 'rt', encoding='utf-8', errors='ignore') as f_in:
    # Create iterator for xml that yields output when tag closes
    elems = (elem for _, elem in ElementTree.iterparse(f_in, events=("end",)))
    # Consume the first element and extract the xml namespace from it. 
    # Although the raw xml has the  short tag names without namespace, i.e. it 
    # has <page> tags and not <http://wwww.mediawiki.org/xml/export...:page> 
    # tags, the parser reads it *with* the namespace. Therefore, it needs the 
    # namespace when looking for child elements in the find function as below.
    elem = next(elems)
    m = re.match("^{(http://www\.mediawiki\.org/xml/export-.*?)}", elem.tag)
    if m is None:
        raise ValueError("Malformed MediaWiki dump")
    ns = {"ns": m.group(1)}
    page_tag = ElementTree.QName(ns['ns'], 'page').text
    # iterate over elements
    for elem in elems:
      if elem.tag == page_tag:
        # Filter out redirect and non-article pages
        if elem.find('./ns:redirect', ns) is not None or \
           elem.find('./ns:ns', ns).text != '0':
          elem.clear()
          continue
        # Extract the article wiki id
        wiki_id = int(elem.find('./ns:id', ns).text)
        # Extract the article title into a variables called title
        title = elem.find('./ns:title', ns).text
        # extract body
        body = elem.find('./ns:revision/ns:text', ns).text

        yield wiki_id, title, body
        elem.clear()

def pages_iter(wiki_file, batch_size=1000):
  """ Generator that yields multiple wiki pages in a batch. Yields the batch 
      index (0, 1, ..) and an iterable of pages with `batch_size` elements. 
      This function is designed to handle batches read directly from the xml file.
  """
  for i, group in groupby(enumerate(page_iter(wiki_file)), 
                          lambda x: x[0] // batch_size):
    _, batch = zip(*group)
    yield i, batch

### CREATE MAPPING OF ARTICLE TITLE TO WIKI ID
def get_title2wid():
  # If file already exists, load it.
  if os.path.exists('title2wid.pkl'):
    with open('title2wid.pkl', 'rb') as f:
      return pickle.load(f)
  # Otherwise, create mapping from scratch 
  RE_wid_title = re.compile(r"\((\d+),0,'(.+?)(?<!\\)','',(0|1)", re.UNICODE)
  # Download wiki pages database
  !wget -N https://dumps.wikimedia.org/enwiki/20210801/enwiki-20210801-page.sql.gz
  wid2title = {}
  title2wid = {}
  re_title2wid = {}
  # wid2re_title = {}
  with gzip.open('enwiki-20210801-page.sql.gz', "rt", encoding='utf-8', errors='ignore') as f:
    for line in f:
      if not line.startswith("INSERT INTO"):
        continue
      for m in RE_wid_title.finditer(line):
        wid = int(m.group(1))
        if m.group(3) == '0':
          wid2title[wid] = m.group(2).lower()
          title2wid[m.group(2).lower()] = wid
        else:
          re_title2wid[m.group(2).lower()] = wid
  # Download wiki page redirect database
  !wget -N https://dumps.wikimedia.org/enwiki/20210801/enwiki-20210801-redirect.sql.gz
  RE_redirects = re.compile(r"\((\d+),0,'(.+?)(?<!\\)'", re.UNICODE)
  wid2target = {}
  with gzip.open('enwiki-20210801-redirect.sql.gz', "rt", encoding='utf-8', errors='ignore') as f:
    for line in f:
      if not line.startswith("INSERT INTO"):
        continue
      for m in RE_redirects.finditer(line):
        wid = int(m.group(1))
        wid2target[wid] = m.group(2).lower()
  # follow redirects
  for re_title, re_id in re_title2wid.items():
    target = wid2target.get(re_id, None)
    if target is None:
      continue
    target_id = title2wid.get(target, None)
    if target_id is None:
      continue
    title2wid[re_title] = target_id

  with open('title2wid.pkl', 'wb') as f:
    pickle.dump(title2wid, f)
  return title2wid

title2wid = get_title2wid()
RE_FORBIDDEN_CHARS = re.compile(r"[#\<\>\[\]\{\}\|]", re.UNICODE)
def get_wiki_id(title):
  """ Return the Wikipedia article id for a given article title or None 
      otherwise.
  """  
  t_new = title.lower().replace(' ', '_').replace("'", "\\'")
  t_new = RE_FORBIDDEN_CHARS.sub('', t_new)
  return title2wid.get(t_new, None)

RE_NON_ARTICLE = re.compile(
  r'(#|:|{|([fF]ile|[iI]mage|[mM]edia|[sS]pecial|[cC]ategory):)', 
  re.UNICODE)
def get_wikilinks(wikicode):
  """ Traverses the parse tree for internal links and filter out non-article 
      links.
  Parameters:
  -----------
  wikicode: mwp.wikicode.Wikicode
    Parse tree of some WikiMedia markdown.
  Returns:
  --------
  list of (target_id: int, anchor_text: str) pairs
    A list of outgoing links from the markdown to wikipedia articles.
  """
  links = []
  for wl in wikicode.ifilter_wikilinks():
    # skip links that don't pass our filter
    title = str(wl.title)
    if RE_NON_ARTICLE.match(title):
      continue
    # remove any lingering section/anchor reference in the link
    title = title.split('#')[0]
    # Get article id from title
    target_id = get_wiki_id(title)
    if target_id is None:
      continue
    # if text is None use title, otherwise strip markdown from the anchor text.
    text = wl.text
    if text is None:
      text = title
    else:
      text = text.strip_code()
    links.append((target_id, text))
  return links

def parse_mediawiki(body):
  wikicode = mwp.parse(body, skip_style_tags=True)
  links = get_wikilinks(wikicode)
  return wikicode.strip_code(), links

# Create preprocessed file
files_data = []
counter = 0
for wiki_id, title, body in page_iter(wiki_file):
  if counter==2000:
    break
  counter+=1
  body, links = parse_mediawiki(body)
  file = (wiki_id, title, body, links)
  files_data.append(file)

with open("part15_preprocessed.pkl", 'wb') as f:
  pickle.dump(files_data, f)