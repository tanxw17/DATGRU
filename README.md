EMNLP 2019 short paper

Recognizing Conflict Opinions in Aspect-level Sentiment Classification with Dual Attention Networks

Code is edited based on https://github.com/wxue004cs/GCAE

Environment:
Python  3.6
Pytorch 0.4.0
torchtext   0.3.1
numpy 1.16.2
sacremoses  0.0.7
nltk    3.3.0

Pretrained Word Embedding:
download the file glove.840B.300d.txt and put it under /embedding folder
http://nlp.stanford.edu/data/glove.840B.300d.zip

Get word_list:

```python
#encoding: utf-8
import os
def get_glove_words(glove_file_path):
    if not os.path.exists(glove_file_path):
        print(glove_file_path + ' not exists!')
        return []
    words_list = []
    with open(glove_file_path, 'r', encoding="utf-8") as fo:
        for line in fo:
            word = line.encode().decode('utf-8').strip().split(' ')[0]
            #print(word)
            words_list.append(word)
    return words_list
```

Produce "glove_words.txt" file in /embedding folder:

```python
words_list = get_glove_words('glove.840B.300d.txt')
print(len(words_list))
with open('glove_words.txt', 'w', encoding="utf-8") as fo:
    for w in words_list:
        fo.write(w+'\n')
```

Run:
Enter /model_files/ 
run train_gcae.sh
run train_ataelstm.sh
run train_datgru.sh

If our code helps, please cite:
```
@inproceedings{tan-etal-2019-recognizing,
    title = "Recognizing Conflict Opinions in Aspect-level Sentiment Classification with Dual Attention Networks",
    author = "Tan, Xingwei  and
      Cai, Yi  and
      Zhu, Changxi",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1342",
    doi = "10.18653/v1/D19-1342",
    pages = "3417--3422",
}
```
