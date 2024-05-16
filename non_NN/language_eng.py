import mmap
import time
import numpy as np
import argparse
import sys
import codecs
from binary_logistic_minibatch import BinaryLogisticRegression
from scipy.spatial import distance


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
    MAX_CACHE_SIZE = 10000

    def mmap_read_word_vectors(fname):
        # Don't forget to close both when done with them
        file_obj = open(fname, mode="r", encoding="utf-8")
        mmap_obj = mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ)
        return mmap_obj, file_obj


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
        dataset = SentenceEquivalence.Dataset()
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f.readlines():
                field = line.strip().split(',')
                if len(field) == 6:
                    self.process_data(dataset, field[3], field[4], field[5])
                else:
                    pass # here there is a coma in the sentence (i would assume)
            return dataset
        return None


    def process_data(self, dataset, sentence1, sentence2, label):
        """
        Processes one line (= one datapoint) in the input file to handle sentence equivalence.
        """
        vector1 = self.compute_sentence_vector(sentence1)
        vector2 = self.compute_sentence_vector(sentence2)
        if len(vector1) > 0 and len(vector2)>0:

            if self.sum_word_vectors:
                vector1 = np.sum(vector1, axis=0)    
                vector2 = np.sum(vector2, axis=0)
                combined_vector = np.concatenate((vector1, vector2))

            elif self.cos_sim_word_vectors:
                combined_vector = self.cosine(vector1, vector2) 
                if not combined_vector:
                    return
            elif self.mincos_word_vectors:
                combined_vector = self.minimum_cosine(vector1, vector2)
                # Steg 2: Lägg till elif statement här           
            else:
                print("in here")
        else:
            return

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
            if self.lowercase_fallback:
                token = token.lower()
            idx = self.w2id.get(token, None)
            if idx is not None:
                vectors.append(self.vec_cache[idx])
            else:
                p = self.pos_cache.get(token)
                if not p:
                    p = self.mo.find("\n{}".format(token).encode('utf8')) + 1
                    self.pos_cache[token] = p
                # normally find returns -1 if not found, but here we have +1

                if p > 0:
                    self.mo.seek(p)
                    line = self.mo.readline()
                    vec = list(map(float, line.decode('utf8').split()[1:]))
                    if self.current_token_id < SentenceEquivalence.MAX_CACHE_SIZE:
                        self.w2id[token] = self.current_token_id
                        self.vec_cache[self.current_token_id, :] = vec
                        self.current_token_id += 1
                    vectors.append(vec)
                else:
                    # Handle missing vectors, use a zero vector
                    vectors.append(np.zeros(self.D))
        # return the word vectors for the whole sentence
        return vectors


    def cosine(self, vector1, vector2):
        """
        Takes the word vectors for each word in two sentences.

        Calculates cosine similarity between each pair of words in the 2 sentences.

        Returns: a list of length len(sen1)*len(sen2). 
        """
        #Setting a limit to how long the cosine vector can be to have them all equal (maybe not ideal)
        if len(vector1)*len(vector2)<400: 
            self.sentenceunder200+=1
            cos_sim_list = []
            for vec1 in vector1:
                for vec2 in vector2:
                    cosine_sim = 1 - distance.cosine(vec1, vec2)
                    cos_sim_list.append(cosine_sim)
            while len(cos_sim_list)<400:
                cos_sim_list.append(0)
            return cos_sim_list        
        else:
            self.sentenceover200 +=1
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
        while len(cos_sim_list)<77:
            cos_sim_list.append(0)
        return cos_sim_list        
    def new_sentence_vector_similarity(self, vector1, vector2):
        # Steg 3 implementera den här funktionen
        pass
        
        # ----------------------------------------------------------

        

    def __init__(self, training_file, test_file, model_file, word_vectors_file,
                 stochastic_gradient_descent, minibatch_gradient_descent, lowercase_fallback,
                 sum_word_vectors, cos_sim_word_vectors, mincos_word_vectors):
        """
        Constructor. Trains and tests a SentenceEquivalence model using binary logistic regression.
        """
        if training_file:
            training_file = 'data/q_pair_train.csv'
        self.sentenceover200=0
        self.sentenceunder200=0
        self.longest_sentence=77
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
            # Train a model
            print("Processing Training file...")
            training_set = self.read_and_process_data(training_file)
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
        if self.sum_word_vectors:
            save_dir = "model_params_sum.txt"
        elif self.cos_sim_word_vectors:
            save_dir = "model_params_cos.txt"
        elif self.mincos_word_vectors:
            save_dir = "model_params_min_cos.txt"
            #Steg 4 lägg till save dir för den nya metoden
        else:
            print("ERROR, method must be entered")
            return
        # Test the model on a test set
        print("Processing test file")
        test_set = self.read_and_process_data(test_file)
        print("Finished processing test file")
        if test_set:
            b.classify_datapoints(test_set.x, test_set.y, save_dir=save_dir)


    # ----------------------------------------------------------

def main():
    """
    Main method. Decodes command-line arguments, and starts the Named Entity Recognition.
    """

    parser = argparse.ArgumentParser(description='Named Entity Recognition', usage='\n* If the -d and -t are both given, the program will train a model, and apply it to the test file. \n* If only -t and -m are given, the program will read the model from the model file, and apply it to the test file.')

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-t', type=str,  required=False, default="data/q_pair_test.csv", help='test file (mandatory)')
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
    #Steg 1. lägg till argument här

    parser.add_argument('-lcf', '--lowercase-fallback', action='store_true')


    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()

    arguments = parser.parse_args()

    sed = SentenceEquivalence(arguments.d, arguments.t, arguments.m, arguments.w, arguments.s, arguments.mgd, arguments.lowercase_fallback, arguments.sum, arguments.cos, arguments.mincos)
    sed.mo.close()
    sed.fo.close()

    input("Press Return to finish the program...")


if __name__ == '__main__':
    main()
