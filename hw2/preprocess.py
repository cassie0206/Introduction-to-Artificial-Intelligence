from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
import string
import re
import nltk
#nltk.download('stopwords')


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    
    
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
       
    preprocessed_text = remove_stopwords(text)

    # Begin your code (Part 0)
    
    def function_punctuation(text: str) -> str:
    
        removed_list = ['!','"','#','$','%','\\','(',')','*','+',',','-','.','..','...','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~','\'','br']
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token.lower() not in removed_list]
        preprocessed_text = ' '.join(filtered_tokens)

        return preprocessed_text
    
    def function_stem(text: str) -> str:
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [PorterStemmer().stem(token) for token in tokens]    
        preprocessed_text = ' '.join(filtered_tokens)
        
        return preprocessed_text
    
    def function_digits(text: str) -> str:
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token[0] not in string.digits]    
        preprocessed_text = ' '.join(filtered_tokens)
        

        return preprocessed_text
    
    def function_lowercase(text: str) -> str:
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = []
        for token in tokens:
            #print(token)
            if len(token) > 1 and token[1:].islower() == False:
                filtered_tokens.append(token.lower())
            else:
                filtered_tokens.append(token)
                 
        preprocessed_text = ' '.join(filtered_tokens)
        
        return preprocessed_text
    
    preprocessed_text = (function_stem(function_digits(function_punctuation(function_lowercase(preprocessed_text)))))
   
    
    # End your code
    #print(preprocessed_text)
    return preprocessed_text

#print('text:')
#preprocessing_function('A great Bugs Bunny cartoon from the earlier years has Bugs as a performer in an window display at a local department store. After he\'s done for the day the manager comes in to tell him that he\'ll be transferring soon. Bugs is happy to oblige into he figures out that the new job is in taxidermy...and that taxidermy has to do with stuffing animals. Animals like say, a certain rabbit. This causes a battle of wits between the rascally rabbit and his now former employer. I found this short to be delightful and definitely one of the better ones of the early 1940\'s. It still remains as funny nearly 60+ years later. This animated short can be seen on Disc 1 of the Looney Tunes Golden Collection Volume 2.<br /><br />My Grade: A-')

