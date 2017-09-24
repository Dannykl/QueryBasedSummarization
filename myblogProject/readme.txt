The code ideas of this work is to integrate the query focus multi document summarization system with Django application. 
The methods that are used in the summarization is lised below:
1-Query likelihood language modelThe summarization system- this is used to retrieve the documents that are relevant to the query
2-Data processing such as query expansion and document segmenting
3-feature extaction such as determine TF-IDF, POS tagging, etc
3-compute the similarity metrics between the query and sentences, and determine the relevant sentences to the query. This is used TF-IDF approaches
4-skfuzzy is used as the final sentence extracter
5-This work was evaluated using ROUGE, and achieves reasonable outcome