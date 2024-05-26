import mmap
import time
import numpy as np
import argparse
import sys
import codecs
from binary_logistic_minibatch import BinaryLogisticRegression
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import csv
import time
import pickle



"""
This is based on the SentenceEquivalence Classifier from assignment 3 
"""
class SentenceEquivalence(object):
    """
    This class performs Semantic equivalence detection (SED).

    It  builds a binary SED model (which distinguishes
    between 'same' or 'different') from training data

    Each line in the data files is supposed to have 3 fields:
    Sentence1, Sentence2, Label

    The 'label' is 'O' if the sentences ask different things
    and '1' if they ask the same thing
    """

    class Dataset(object):
        """
        Internal class for representing a dataset.
        """
        def __init__(self):

            #  The list of datapoints. Each datapoint is itself
            #  a list of features (each feature coded as a number).
            self.x = []

            #  The list of labels for each datapoint. The datapoints should
            #  have the same order as in the 'x' list above.
            self.y = []

    # --------------------------------------------------

    """
    Word vector feature computation
    """
    PAD_SYMBOL = "<pad>"
    MAX_CACHE_SIZE = 1000000

    def mmap_read_word_vectors(fname):
        # Don't forget to close both when done with them
        file_obj = open(fname, mode="r", encoding="utf-8")
        mmap_obj = mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ)
        return mmap_obj, file_obj

    def load_cache(self, filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_cache(self, cache, filename):
        with open(filename, 'wb') as f:
            pickle.dump(cache, f)

    def own_features():
        pass # Could manually add features, but probably won't

    class FeatureFunction(object):
        def __init__(self, func, boolean=True):
            self.func = func
            self.boolean = boolean

        def evaluate(self):
            if self.boolean:
                return 1 if self.func() else 0
            else:
                return self.func()



    # --------------------------------------------------

    def label_number(self, s):
        return 0 if '0' == s else 1

    def read_and_process_data(self, filename):
        """
        Read the input file and return the dataset.
        """
        if self.weighted_avg_word_vectors:
            self.fitting_TFIDF(filename)
        
        dataset = SentenceEquivalence.Dataset()
        corpus = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            reader = csv.reader(f)
            count=0
            old_time = 0
            start=time.time()
            for line in reader:
                count+=1

                if count%10000 == 0:
                    end= time.time()
                    if start:
                        print(f"length of cache {len(self.pos_cache)}")
                        print(f"Time elapsed {end-start-old_time}")
                        print(f"number of sentences: {count}")
                        old_time = end-start

                if len(line) == 3:

                    field0 = self.clean_and_tokenize(line[0])
                    field1 = self.clean_and_tokenize(line[1])
                    corpus.append(field0)
                    corpus.append(field1)
                    self.process_data(dataset, field0, field1, line[2])
                else:
                    print("ERROR, length of line != 3")
                    pass # We never get here

            return dataset
        return None

    def fitting_TFIDF(self, filename):
        """
        Read the input file and return the dataset.
        """
        
        dataset = SentenceEquivalence.Dataset()
        corpus = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            reader = csv.reader(f)
            count=0
            start = time.time()
            for line in reader:
                count+=1
                if count%10000 == 0:
                    print(f"sentence number: {count}")
                if len(line) == 3:
                    field0 = self.clean_and_tokenize(line[0])
                    field1 = self.clean_and_tokenize(line[1])
                    corpus.append(field0)
                    corpus.append(field1)
                else:
                    pass
            end = time.time()
            print(f"Time to get corpus: {end - start}")
            print(corpus[0:20])
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            print(f"self.tfidf_matrix.shape: {self.tfidf_matrix.shape}")
        return None

    def clean_and_tokenize(self, text):
        text = text.lower()
        text = re.findall(r'\b\w+\b', text)
        text = ' '.join(text)
        return text

    def process_data(self, dataset, sentence1, sentence2, label):
        """
        Processes one line (= one datapoint) in the input file to handle sentence equivalence.
        """
        start = time.time()
        vector1 = self.compute_sentence_vector(sentence1)
        vector2 = self.compute_sentence_vector(sentence2)
        
        if self.weighted_avg_word_vectors:
            vector1 = self.compute_weighted(sentence1, vector1)
            vector2 = self.compute_weighted(sentence2, vector2)
        if len(vector1) > 0 and len(vector2)>0:
            if self.sum_word_vectors:
                vector1 = np.sum(vector1, axis=0)/len(sentence1)  
                vector2 = np.sum(vector2, axis=0)/len(sentence1)
                difference_v = abs(vector1 - vector2)
                combined_vector = np.concatenate((vector1, vector2))
                combined_vector = np.concatenate((combined_vector, difference_v))

            elif self.cos_sim_word_vectors:
                combined_vector = self.cosine(vector1, vector2) 
                combined_vector = np.sort(combined_vector)

            elif self.mincos_word_vectors:
                combined_vector = self.minimum_cosine(vector1, vector2)

            else:
                return
        else:
            return
        end = time.time()
        #print(f"process_data: {end - start}")
        dataset.x.append(combined_vector)
        dataset.y.append(self.label_number(label))


    def read_model(self, filename):
        """
        Read a model from file
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            d = map(float, f.read().split(' '))
        return list(d)

    def compute_sentence_vector(self, sentence):
        words = sentence.split()
        vectors = []
        for token in words:
            start = time.time()
            if self.lowercase_fallback:
                token = token.lower()
            idx = self.w2id.get(token, None)
            if idx is not None:
                vectors.append(self.vec_cache[idx])
                end = time.time()
            else:
                self.i+=1
                if self.i <10000000:
                    p = self.pos_cache.get(token)
                    if not p:
                        self.mo.seek(0) # reset  
                        start=time.time()
                        chunk_size = 10000  # size of chunks to read
                        while time.time()-start<0.5:
                            
                            curr = self.mo.tell()
                            chunk = self.mo.read(chunk_size)    
                            if not chunk:
                                break                    
                            p = chunk.find("\n{}".format(token).encode('utf8')) + 1
                            if p != 0:
                                p = curr + p
                                break

                        #print(f"p for 'the' {p}")
                        self.pos_cache[token] = p 
                    # normally find returns -1 if not found, but here we have +1
                    if p > 0:
                        self.mo.seek(p)
                        line = self.mo.readline()
                        vec = list(map(float, line.decode('utf8').split()[1:]))
                        #end = time.time()
                        #print(f"Time when word added to cache {start-end}")
                    else:
                        # Handle missing vectors, use a zero vector
                        end = time.time()
                        vec = np.zeros(self.D)
                        #print(f"Time when word not found {end-start}")
                    vectors.append(vec)
                    #print(f"vectors_len{len(vectors)}")
                else:
                    vec = np.zeros(self.D)
                    vectors.append(np.zeros(self.D)) 
                if self.current_token_id < SentenceEquivalence.MAX_CACHE_SIZE:

                    self.w2id[token] = self.current_token_id
                    self.vec_cache[self.current_token_id, :] = vec
                    self.current_token_id += 1
                else:
                    print("error by max cache size")

                # return the word vectors for the whole sentence
                end = time.time()
                #print(f"compute_sentence_vector:{end-start}")

        return vectors

    def compute_weighted(self, sentence, vector):
        words = sentence.split()
        sentence_length = len(words)
        tfidf_scores = self.tfidf_vectorizer.transform([' '.join(words)]).toarray()[0]
        weights = []
        for word in words:
            tfidf_index = self.tfidf_vectorizer.vocabulary_.get(word)
            if tfidf_index is not None:
                weights.append(tfidf_scores[tfidf_index])
            else:
                weights.append(0.0)  # If the word is not found in the TF-IDF vectorizer, assign zero weight
        weights = np.array(weights)
        vector = np.array(vector)
        for word_vec_i in range(len(vector)):
            vector[word_vec_i] = weights[word_vec_i]*vector[word_vec_i]        
        return vector

    def cosine(self, vector1, vector2):
        """
        Takes the word vectors for each word in two sentences.

        Calculates cosine similarity between each pair of words in the 2 sentences.

        Returns: a list of length len(sen1)*len(sen2). 
        """
        #Setting a limit to how long the cosine vector can be to have them all equal (maybe not ideal)
        product_length = len(vector1)*len(vector2)
        if product_length<400: 
            cos_sim_list = []
            for vec1 in vector1:
                for vec2 in vector2:
                    cosine_sim = 1 - distance.cosine(vec1, vec2)
                    cos_sim_list.append(cosine_sim)
            #cos_sim_list = [cos_sim_list[i]/(product_length) for i in range(product_length)]  

            while len(cos_sim_list)<400:
                cos_sim_list.append(0)
            return cos_sim_list     
        else:
            return np.zeros(400)
            #print(f"sentenceover200 {self.sentenceover200} sentence under200 {self.sentenceunder200}")


    def minimum_cosine(self, vector1, vector2):
        """
        For each word in sentence 1 takes the cosine similarity between all words in sentence 2
        then appends the minimum cosine similarity for each word in sentence 1.

        Returns: a list of length len(sen1). 
        """
        cos_sim_list = []
        for vec1 in vector1:
            cos_for_index = []
            for vec2 in vector2:
                cosine_sim = 1 - distance.cosine(vec1, vec2)
                cos_for_index.append(cosine_sim)
            cos_sim_list.append(min(cos_for_index))
            cos_sim_list.append(np.mean(cos_for_index))
            cos_sim_list.append(max(cos_for_index))
        while len(cos_sim_list)<3*self.longest_sentence:
            cos_sim_list.append(0)
        if len(cos_sim_list)!=3*self.longest_sentence:
            #print(f"lencossimlist{len(cos_sim_list)}")
            return np.zeros(3*self.longest_sentence)
        return cos_sim_list  

    def new_sentence_vector_similarity(self, vector1, vector2):
        # Steg 3 implementera den här funktionen
        pass
        
        # ----------------------------------------------------------
    def input_sentence(self, sentence1, sentence2):
        sentence_1 = self.clean_and_tokenize(sentence1)
        sentence_2 = self.clean_and_tokenize(sentence2)

        vector1 = self.compute_sentence_vector(sentence1)
        vector2 = self.compute_sentence_vector(sentence2)
        vector1 = np.sum(vector1, axis=0)/len(sentence1)  
        vector2 = np.sum(vector2, axis=0)/len(sentence1)
        difference_v = abs(vector1 - vector2)
        combined_vector = np.concatenate((vector1, vector2))
        combined_vector = np.concatenate((combined_vector, difference_v))
        combined_vector = np.concatenate(([1],combined_vector)) #adding bias
        return combined_vector


    def __init__(self, training_file, test_file, model_file, word_vectors_file,
                 stochastic_gradient_descent, minibatch_gradient_descent, lowercase_fallback,
                 sum_word_vectors, cos_sim_word_vectors, mincos_word_vectors, weighted_avg_word_vectors,
                 loadcache):
        """
        Constructor. Trains and tests a SentenceEquivalence model using binary logistic regression.
        """
        if training_file:
            training_file = 'data/train.csv'


        # Initialize TF-IDF vectorizer and compute TF-IDF matrix for the corpus
        self.tfidf_vectorizer = TfidfVectorizer()
        self.i = 0
        self.loadcache=loadcache
        self.longest_sentence=78
        self.weighted_avg_word_vectors = weighted_avg_word_vectors # this is tfidf True/False

        self.mincos_word_vectors = mincos_word_vectors
        self.lowercase_fallback = lowercase_fallback
        self.sum_word_vectors = sum_word_vectors
        self.cos_sim_word_vectors = cos_sim_word_vectors
        self.current_token = None #  The token currently under consideration.
        self.last_token = None #  The token on the preceding line.

        # self.W, self.i2w, self.w2i = SentenceEquivalence.read_word_vectors(word_vectors_file)
        self.mo, self.fo = SentenceEquivalence.mmap_read_word_vectors(word_vectors_file)
        # get the dimensionality of the vectors
        p = self.mo.find("\nthe".encode('utf8')) + 1
        self.mo.seek(p)
        line = self.mo.readline()
        vec = list(map(float, line.decode('utf8').split()[1:]))
        self.D = len(vec)

        self.current_token_id = 0
        self.pos_cache, self.w2id, self.vec_cache = {}, {}, np.zeros((SentenceEquivalence.MAX_CACHE_SIZE, self.D))
        self.w2id["the"] = self.current_token_id
        self.vec_cache[self.current_token_id,:] = vec
        self.current_token_id += 1

        p = self.mo.find(SentenceEquivalence.PAD_SYMBOL.encode('utf8'))

        self.w2id[SentenceEquivalence.PAD_SYMBOL] = self.current_token_id
        self.vec_cache[self.current_token_id,:] = vec
        self.current_token_id += 1

       

        if training_file:
            if self.loadcache:
                # Loads these from files, much faster because the .find call in create sentence vectors is very slow 
                self.pos_cache = {}
                self.vec_cache = np.zeros((SentenceEquivalence.MAX_CACHE_SIZE, self.D))
                self.w2id = {}
                self.pos_cache = self.load_cache('pos_cache.pkl')
                self.vec_cache = self.load_cache('vector_cache.pkl')
                self.w2id = self.load_cache('w2id.pkl')
            
            # Train a model
            print("Processing Training file...")
            training_set = self.read_and_process_data(training_file)
            if not self.loadcache:
                self.save_cache(self.pos_cache, 'pos_cache.pkl')
                self.save_cache(self.vec_cache, 'vector_cache.pkl')
                self.save_cache(self.w2id,'w2id.pkl')


            if training_set:
                print("Training Model... ")

                start_time = time.time()
                b = BinaryLogisticRegression(training_set.x, training_set.y)
                if stochastic_gradient_descent:
                    b.stochastic_fit_with_early_stopping()
                elif minibatch_gradient_descent:
                    print("Starting minibatch...")
                    b.minibatch_fit_with_early_stopping()
                else:
                    b.fit()
                print("Model training took {}s".format(round(time.time() - start_time, 2)))


        else:
            print("Opening Model")
            model = self.read_model(model_file)
            if model:
                b = BinaryLogisticRegression(theta=model)
        
        # Saving the trained model
        if self.sum_word_vectors and not self.weighted_avg_word_vectors:
            save_dir = "model_params_sum"
        elif self.cos_sim_word_vectors and not self.weighted_avg_word_vectors:
            save_dir = "model_params_cos"
        elif self.mincos_word_vectors and not self.weighted_avg_word_vectors:
            save_dir = "model_params_min_cos"
        elif self.weighted_avg_word_vectors and sum_word_vectors:  # Save directory for the new method
            save_dir = "model_params_sum_tfidf"
        elif self.weighted_avg_word_vectors and self.cos_sim_word_vectors:
            save_dir = "model_params_cos_tfidf"
        elif self.weighted_avg_word_vectors and self.mincos_word_vectors:
            save_dir = "model_params_mincos_tfidf"
        else:
            print("ERROR, method must be entered")
            return
        # Test the model on a test set
        print("Processing test file")

        self.pos_cache = {}
        self.vec_cache = np.zeros((SentenceEquivalence.MAX_CACHE_SIZE, self.D))
        self.w2id = {}
        if self.loadcache and test_file == "data/test.csv":
            print("yes")
            # Loads these from files, much faster because the .find call in create sentence vectors is very slow 
            self.pos_cache = self.load_cache('pos_cache_test.pkl')
            self.vec_cache = self.load_cache('vector_cache_test.pkl')
            self.w2id = self.load_cache('w2id_test.pkl')
            print(f"length of cache: {len(self.pos_cache)}")

                
        test_set = self.read_and_process_data(test_file)
        if not self.loadcache:
            self.save_cache(self.pos_cache, "pos_cache_test.pkl")
            self.save_cache(self.vec_cache, "vector_cache_test.pkl")
            self.save_cache(self.w2id, "w2id_test.pkl")

        print("Finished processing test file")
        if test_set:
            b.classify_datapoints(test_set.x, test_set.y, save_dir=save_dir)
        sentence1, sentence2 = "", ""

        while sentence1 != "f" or sentence2!="f":
            print("Enter f to finish the program")
            sentence1 = input("Enter the first sentence: ")
            sentence2 = input("Enter the second sentence: ")
            feature_vector = self.input_sentence(sentence1, sentence2)
            print(len(feature_vector))
            prob = b.sigmoid(np.dot(feature_vector, b.theta))
            if prob>0.5:
                label="the same thing!"
            else:
                label="different things!"
            print(f"The model predicts the sentences to ask: {label}")
    # ----------------------------------------------------------

def main():
    """
    Main method. Decodes command-line arguments, and starts the Named Entity Recognition.
    """

    parser = argparse.ArgumentParser(description='Named Entity Recognition', usage='\n* Give method (-sum/-cos/-mincos/-tfidf) and -d or -m')

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-t', type=str,  required=False, default="data/test.csv", help='test file (mandatory)')
    required_named.add_argument('-w', type=str, required=False, default="data/en.vectors", help='file with word vectors')

    group = required_named.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', action='store_true', help='training file (required if -m is not set)')
    group.add_argument('-m', type=str, help='model file (required if -d is not set)')

    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument('-s', action='store_true', default=False, help='Use stochastic gradient descent')
    group2.add_argument('-b', action='store_true', default=False, help='Use batch gradient descent')
    group2.add_argument('-mgd', action='store_true', default=True, help='Use mini-batch gradient descent')

    # New arguments to choose way of representing the sentences 
    group3 = parser.add_mutually_exclusive_group(required=True)
    group3.add_argument('-sum', action='store_true', default=False, help='Use a sum of the word vectors ')
    group3.add_argument('-cos', action='store_true', default=False, help='Use a list of all cosine similarities')
    group3.add_argument('-mincos', action='store_true', default=False, help='Use a list of all minimum cosine similarities')
    #Step 1: add arguments for other metrics

    group4 = parser.add_mutually_exclusive_group(required=False)
    group4.add_argument('-tfidf', action='store_true', default=False, help='Use a tf-idf weight on the word embeddings')

    group4 = parser.add_mutually_exclusive_group(required=False)
    group4.add_argument('-loadcache', action='store_true', default=False, help='Loads cache for word vectors (much much faster, but requires the file)')

    parser.add_argument('-lcf', '--lowercase-fallback', action='store_true')


    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()

    arguments = parser.parse_args()

    sed = SentenceEquivalence(arguments.d, arguments.t, arguments.m, arguments.w, arguments.s, arguments.mgd, arguments.lowercase_fallback, arguments.sum, arguments.cos, arguments.mincos, arguments.tfidf, arguments.loadcache)
    sed.mo.close()
    sed.fo.close()

    input("Press Return to finish the program...")


if __name__ == '__main__':
    main()
