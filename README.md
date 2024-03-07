cd the root of the directory

Install the relevant packages required for the program to run

Run python bbc_news_classification.py in root of the directory


Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
Natural Classes: 5 (business, entertainment, politics, sport, tech)

If you make use of the dataset, please consider citing the publication: 
- D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

All rights, including copyright, in the content of the original articles are owned by the BBC.

Contact Derek Greene <derek.greene@ucd.ie> for further information.
http://mlg.ucd.ie/datasets/bbc.html


Introduction
In text classification tasks, understanding the distributional characteristics of the data is crucial as it significantly influences various aspects of the modeling process, including the choice of model architecture, preprocessing techniques, and feature selection strategies. 
Analyzing the distributional characteristics of the BBC_news dataset offers valuable insights into its composition and structure. Here are some key aspects to consider:
![image](https://github.com/MentalF1ow/Applications-of-Machine-Learning/assets/162504809/9e59f875-3685-4649-9915-a079da2a55eb)

Assessing the distribution of documents across different categories reveals whether the dataset suffers from class imbalances. The above figure shows the distribution of each category. It can be found that the distribution of each label is relatively uniform, and there is no sample imbalance. Therefore, the common proportion division can be used when dividing the data set, and the ratio of training:test = 8:2 is also used in this code.
![image](https://github.com/MentalF1ow/Applications-of-Machine-Learning/assets/162504809/30ebe27d-2371-476c-8157-b23135d2ed12)

Preprocessing
Since the original data may contain a large number of missing values, noise, outliers, or duplicate values. Before delving into preprocessing, it's essential to assess the state of the raw data. This evaluation typically reveals a myriad of imperfections that could potentially undermine subsequent analysis tasks. Here are the results before and after preprocessing the raw data to remove non-letter characters, stop words, and extra Spaces.

Feature selection
1.Word Frequency Features (CountVectorizer)
2.TF-IDF Features (TfidfVectorizer)
3.Word Embeddings Features (Word2Vec)

Evaluation
	Three features were selected, so the performance of each feature was first verified separately. In the table below are the results of the experiment. It can be found that among the three features of Word Frequency, TF-IDF and Word Embeddings, Word Frequency and TF-IDF perform well in the current task, achieving an accuracy of 0.95, while Word Embeddings has the worst ability. This is only 0.79 accuracy.

Word Frequency	
![image](https://github.com/MentalF1ow/Applications-of-Machine-Learning/assets/162504809/0e2b76a4-342c-4fbc-8767-e2ddb52f08bf)

TF-IDF	
![image](https://github.com/MentalF1ow/Applications-of-Machine-Learning/assets/162504809/9bb56772-1f0a-4505-b400-dcdaecc95ec5)

Word Embeddings	
![image](https://github.com/MentalF1ow/Applications-of-Machine-Learning/assets/162504809/4d0ddc03-6e22-496e-a197-c3a9dd2df7b4)
