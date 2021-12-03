<<<<<<< HEAD
<<<<<<< HEAD
class TopicModel():

    def __init__(self,model):
        self.model = model
        
    def _extractNewsContent(self, url):

        '''Apply this function to a column of G-delt news Urls;
           the function extracts the news articles from the Urls,
           and returns text corpus
        '''
        from newspaper import Article
        import pandas as pd
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            news_text =  ''.join(text)
        except Exception:
            return ""

        return news_text

    def _clean_data(self, texts):

        import re
        words = list()
        for text in texts.split():
            # remove web Urls
            text = re.sub(r'https?[://a-zA-Z\.0-9]*','', text)
            text = re.sub(r'https?://(www\.)?(\w+)(\.\w+)',r'',text)
            # remove non text character from start and end of string
            text = re.sub(r'(^\W+|\W+$)','',text)
            # remove multiple white spaces
            text = re.sub(r'\s+',' ',text)
            #remove non text characters and emojis between texts
            text = re.sub(r'\W+',r' ',text)
            #remove white space at the end of strings
            text = re.sub(r'\s+$',r'',text)
            # Remove unwanted symbols
            text = re.sub(r'[#,@,$_,?*//""]',r'',text)

            words.append(text.lower())

        return " ".join(words)

    def _special_lemmatizer(self, text):
        import spacy
        allowed_postags=['ADJ','VERB','NOUN','ADV']
        nlp = spacy.load('en_core_web_sm')
        token = [word for word in nlp(text) if word.pos_ in allowed_postags]
        token1 = [word.lemma_ for word in token]
        token2 = [word for word in token1]
        return ' '.join(token2)

    def _ner_model(self,url,ner):
        import spacy
        import pandas as pd
        ner_news = self._extractNewsContent(url)
        model = spacy.load(ner)
        text = model(ner_news)
        location = set()
        date = set()
        for ent in text.ents:
            if ent.label_ == 'Date':
                date.add(str(ent))
            else:
                location.add(str(ent))
        loca = [str(i) for i in location]
        date_ = [str(i) for i in date]
        join_loca = [' '.join(loca)]
        join_date = [' '.join(date_)]
        df = pd.DataFrame({'Location':join_loca, 'Date':join_date})
        
        return df


    def predict(self,url,ner):
        import pandas as pd
        import numpy as np

        #url = df.Url.iloc[0]
        news = [self._extractNewsContent(url)]
        cleaned = self._clean_data(''.join(news))
        lemma = [self._special_lemmatizer(cleaned)]
        topic_names =['Terrorism/Banditry','Protest','Violence','Terrorism/Banditry','Battle','Assault']
        distribution = self.model.transform(lemma)
        topics = [topic_names[np.argmax(topics)] for topics in distribution]
        #topic_proba = [np.max(proba) for proba in distribution]
        ner_output = self._ner_model(url,ner)
        lda_output = pd.DataFrame(zip(news,topics),columns = ['News','Topic'])
        result = pd.concat([lda_output,ner_output], axis=1)
        return result
=======
python -m spacy download en_core_web_sm

class TopicModel():
    

    def __init__(self,model):
        self.model = model
        
    def _extractNewsContent(self, url):

        '''Apply this function to a column of G-delt news Urls;
           the function extracts the news articles from the Urls,
           and returns text corpus
        '''
        from newspaper import Article
        import pandas as pd
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            news_text =  ''.join(text)
        except Exception:
            return ""

        return news_text

    def _clean_data(self, texts):

        import re
        words = list()
        for text in texts.split():
            # remove web Urls
            text = re.sub(r'https?[://a-zA-Z\.0-9]*','', text)
            text = re.sub(r'https?://(www\.)?(\w+)(\.\w+)',r'',text)
            # remove non text character from start and end of string
            text = re.sub(r'(^\W+|\W+$)','',text)
            # remove multiple white spaces
            text = re.sub(r'\s+',' ',text)
            #remove non text characters and emojis between texts
            text = re.sub(r'\W+',r' ',text)
            #remove white space at the end of strings
            text = re.sub(r'\s+$',r'',text)
            # Remove unwanted symbols
            text = re.sub(r'[#,@,$_,?*//""]',r'',text)

            words.append(text.lower())

        return " ".join(words)

    def _special_lemmatizer(self, text):
        import spacy
        allowed_postags=['ADJ','VERB','NOUN','ADV']
        nlp = spacy.load('en_core_web_sm')
        token = [word for word in nlp(text) if word.pos_ in allowed_postags]
        token1 = [word.lemma_ for word in token]
        token2 = [word for word in token1]
        return ' '.join(token2)

    def _ner_model(self,url,ner):
        import spacy
        import pandas as pd
        ner_news = self._extractNewsContent(url)
        model = spacy.load(ner)
        text = model(ner_news)
        location = set()
        date = set()
        for ent in text.ents:
            if ent.label_ == 'Date':
                date.add(str(ent))
            else:
                location.add(str(ent))
        loca = [str(i) for i in location]
        date_ = [str(i) for i in date]
        join_loca = [' '.join(loca)]
        join_date = [' '.join(date_)]
        df = pd.DataFrame({'Location':join_loca, 'Date':join_date})
        
        return df


    def predict(self,url,ner):
        import pandas as pd
        import numpy as np

        #url = df.Url.iloc[0]
        news = [self._extractNewsContent(url)]
        cleaned = self._clean_data(''.join(news))
        lemma = [self._special_lemmatizer(cleaned)]
        topic_names =['Terrorism/Banditry','Protest','Violence','Terrorism/Banditry','Battle','Assault']
        distribution = self.model.transform(lemma)
        topics = [topic_names[np.argmax(topics)] for topics in distribution]
        #topic_proba = [np.max(proba) for proba in distribution]
        ner_output = self._ner_model(url,ner)
        lda_output = pd.DataFrame(zip(news,topics),columns = ['News','Topic'])
        result = pd.concat([lda_output,ner_output], axis=1)
        return result
>>>>>>> 80042b8b8dc3c80c7e4d82f0e93335b7783a6caa
=======
class TopicModel():
    

    def __init__(self,model):
        self.model = model
        
    def _extractNewsContent(self, url):

        '''Apply this function to a column of G-delt news Urls;
           the function extracts the news articles from the Urls,
           and returns text corpus
        '''
        from newspaper import Article
        import pandas as pd
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            news_text =  ''.join(text)
        except Exception:
            return ""

        return news_text

    def _clean_data(self, texts):

        import re
        words = list()
        for text in texts.split():
            # remove web Urls
            text = re.sub(r'https?[://a-zA-Z\.0-9]*','', text)
            text = re.sub(r'https?://(www\.)?(\w+)(\.\w+)',r'',text)
            # remove non text character from start and end of string
            text = re.sub(r'(^\W+|\W+$)','',text)
            # remove multiple white spaces
            text = re.sub(r'\s+',' ',text)
            #remove non text characters and emojis between texts
            text = re.sub(r'\W+',r' ',text)
            #remove white space at the end of strings
            text = re.sub(r'\s+$',r'',text)
            # Remove unwanted symbols
            text = re.sub(r'[#,@,$_,?*//""]',r'',text)

            words.append(text.lower())

        return " ".join(words)

    def _special_lemmatizer(self, text):
        import spacy
        allowed_postags=['ADJ','VERB','NOUN','ADV']
        nlp = spacy.load('en_core_web_sm')
        token = [word for word in nlp(text) if word.pos_ in allowed_postags]
        token1 = [word.lemma_ for word in token]
        token2 = [word for word in token1]
        return ' '.join(token2)

    def _ner_model(self,url,ner):
        import spacy
        import pandas as pd
        ner_news = self._extractNewsContent(url)
        model = spacy.load(ner)
        text = model(ner_news)
        location = set()
        date = set()
        for ent in text.ents:
            if ent.label_ == 'Date':
                date.add(str(ent))
            else:
                location.add(str(ent))
        loca = [str(i) for i in location]
        date_ = [str(i) for i in date]
        join_loca = [' '.join(loca)]
        join_date = [' '.join(date_)]
        df = pd.DataFrame({'Location':join_loca, 'Date':join_date})
        
        return df


    def predict(self,url,ner):
        import pandas as pd
        import numpy as np

        #url = df.Url.iloc[0]
        news = [self._extractNewsContent(url)]
        cleaned = self._clean_data(''.join(news))
        lemma = [self._special_lemmatizer(cleaned)]
        topic_names =['Terrorism/Banditry','Protest','Violence','Terrorism/Banditry','Battle','Assault']
        distribution = self.model.transform(lemma)
        topics = [topic_names[np.argmax(topics)] for topics in distribution]
        #topic_proba = [np.max(proba) for proba in distribution]
        ner_output = self._ner_model(url,ner)
        lda_output = pd.DataFrame(zip(news,topics),columns = ['News','Topic'])
        result = pd.concat([lda_output,ner_output], axis=1)
        return result
>>>>>>> 871b5ae23766f7c146fcf4f0f1e92738a61ac9b2