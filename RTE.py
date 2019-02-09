from __future__ import print_function

import xlwt
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.classify.util import accuracy, check_megam_config
from nltk.classify.maxent import MaxentClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import rte as rte_corpus
from sklearn.metrics import classification_report


class RTEFeatureExtractor(object):
    """
    This builds a bag of words for both the text and the hypothesis after
    throwing away some stopwords, then calculates overlap and difference.
    """

    def __init__(self, rtepair, stop=True, use_lemmatize=True):
        """
        :param rtepair: a ``RTEPair`` from which features should be extracted
        :param stop: if ``True``, stopwords are thrown away.
        :type stop: bool
        """
        self.stop = stop
        self.stopwords = set(stopwords.words('english'))

        self.negwords = set(['no', 'not', 'never', 'failed', 'rejected', 'denied' , 'nothing'])
        # Try to tokenize so that abbreviations, monetary amounts, email
        # addresses, URLs are single tokens.
        tokenizer = RegexpTokenizer('[\w.@:/]+|\w+|\$[\d.]+')

        # Get the set of word types for text and hypothesis
        # self.text_tokens = tokenizer.tokenize(rtepair.text)
        # self.hyp_tokens = tokenizer.tokenize(rtepair.hyp)

        #
        self.text_tokens = nltk.word_tokenize(rtepair.text)
        self.hyp_tokens = nltk.word_tokenize(rtepair.hyp)

        self.text_words = set(self.text_tokens)
        self.hyp_words = set(self.hyp_tokens)
        self.tagged_words_txt = nltk.pos_tag(self.text_words)
        self.tagged_words_hyp = nltk.pos_tag(self.hyp_words)
        self.ne_text = [x[0] for x in self.extractNE(nltk.ne_chunk(self.tagged_words_txt, binary=True))]
        self.ne_hyp = [x[0] for x in self.extractNE(nltk.ne_chunk(self.tagged_words_hyp, binary=True))]

        if use_lemmatize:
            self.text_words = set(self.lemmatize(token) for token in self.text_tokens)
            self.hyp_words = set(self.lemmatize(token) for token in self.hyp_tokens)

        if self.stop:
            self.text_words = self.text_words - self.stopwords
            self.hyp_words = self.hyp_words - self.stopwords

        self._overlap = self.hyp_words & self.text_words
        self._hyp_extra = self.hyp_words - self.text_words
        self._txt_extra = self.text_words - self.hyp_words

        is_noun = lambda pos: pos[:2] == 'NN'
        is_verb = lambda pos: pos[:2] == 'VB'

        self.txt_nouns = set([word for (word, pos) in nltk.pos_tag(self.text_words) if is_noun(pos)])
        self.txt_verb = set([word for (word, pos) in nltk.pos_tag(self.text_words) if is_verb(pos)])

        self.hyp_nouns = set([word for (word, pos) in nltk.pos_tag(self.hyp_words) if is_noun(pos)])
        self.hyp_verb = set([word for (word, pos) in nltk.pos_tag(self.hyp_words) if is_verb(pos)])

        self._overlap = self.hyp_words & self.text_words

        self._syn_overlap_count = self.synomyms_overlap(self.hyp_words, self.text_words)

        self._overlap_noun = self.txt_nouns & self.hyp_nouns
        self._overlap_verb = self.txt_verb & self.hyp_verb

        self._hyp_extra_noun = self.hyp_nouns - self.txt_nouns
        self._hyp_extra_verb = self.hyp_verb - self.txt_verb

        self._txt_extra_noun = self.txt_nouns - self.hyp_nouns
        self._txt_extra_verb = self.txt_verb - self.hyp_verb

        self._cosine_sim = self.cosine_similarity(self.text_words, self.hyp_words)



    def overlap_new(self, toktype, debug=False):
        """
        Compute the overlap between text and hypothesis.

        :param toktype: distinguish Named Entities from ordinary words
        :type toktype: 'ne' or 'word'
        """
        ne_overlap = set(token for token in self._overlap if self.ne_all(token))
        if toktype == 'ne':
            if debug:
                print("ne overlap", ne_overlap)
            return ne_overlap
        elif toktype == 'word':
            if debug:
                print("word overlap", self._overlap)
            return self._overlap
        elif toktype == 'noun':
            if debug:
                print("word overlap", self._overlap_noun)
            return self._overlap_noun
        elif toktype == 'verb':
            if debug:
                print("word overlap", self._overlap_verb)
            return self._overlap_verb

        else:
            raise ValueError("Type not recognized:'%s'" % toktype)

    def extractNE(self, ne):

        named_entities = []
        for tagged_tree in ne:
            if hasattr(tagged_tree, 'label'):
                entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  #
                entity_type = tagged_tree.label()  # get NE category
                named_entities.append((entity_name, entity_type))
        return (named_entities)

    def hyp_extra_new(self, toktype):
        """
        Compute the extraneous material in the hypothesis.

        :param toktype: distinguish Named Entities from ordinary words
        :type toktype: 'ne' or 'word'
        """

        ne_extra = set(token for token in self._hyp_extra if self._ne_hyp(token))

        if toktype == 'ne':
            return ne_extra
        elif toktype == 'word':
            return self._hyp_extra - ne_extra
        elif toktype == 'noun':
            return self._hyp_extra_noun
        elif toktype == 'verb':
            return self._hyp_extra_verb
        else:
            raise ValueError("Type not recognized: '%s'" % toktype)

    def _ne_hyp(self, token):
        """
        This just assumes that words in all caps or titles are
        named entities.

        :type token: str
        """

        if token.istitle() or token.isupper() or token in self.ne_text:
            # if token in self.ne_text :
            return True
        return False

    def ne_txt(self, token):
        """
        This just assumes that words in all caps or titles are
        named entities.

        :type token: str
        """

        # if token.istitle() or token.isupper() or token in self.ne_text:
        if token in self.ne_text:
            return True
        return False

    def ne_all(self, token):
        """
        This just assumes that words in all caps or titles are
        named entities.

        :type token: str
        """

        # if token.istitle() or token.isupper() or token in self.ne_text:
        if token in self.ne_text or token in self.ne_hyp:
            return True
        return False

    @staticmethod
    def word_meaning(word):
        syns = wordnet.synsets(word)
        meaning = [x.lemmas()[0].name() for x in syns]
        return (meaning)

    @staticmethod
    def synomyms_overlap(list, hypothesis):
        count = 0
        for x in list:
            meaning = RTEFeatureExtractor.word_meaning(x)
            if len(meaning) != 0:
                intersection = set(hypothesis) & set(meaning)
                if len(intersection) > 0:
                    count += 1
        return count

    @staticmethod
    def lemmatize(word):
        """
        Use morphy from WordNet to find the base form of verbs.
        """
        # lmtzr= WordNetLemmatizer()
        # lemma = lmtzr.lemmatize(word)
        lemma = nltk.corpus.wordnet.morphy(word, pos=nltk.corpus.wordnet.VERB)
        # print('lemma')
        if lemma is not None:
            return lemma
        return word

    @staticmethod
    def word_count(words):
        fdist = nltk.FreqDist(words)
        filtered_word_freq = dict((word, freq) for word, freq in fdist.items() if not word.isdigit())
        # print(filtered_word_freq)
        return filtered_word_freq

    @staticmethod
    def cosine_similarity(sent1, sent2):
        word_count1 = RTEFeatureExtractor.word_count(sent1)
        word_count2 = RTEFeatureExtractor.word_count(sent2)
        sent = set(sent2).union(set(sent1))
        sent_freq1 = [word_count1[x] if x in word_count1 else 0 for x in sent]
        sent_freq2 = [word_count2[x] if x in word_count2 else 0 for x in sent]
        num = sum(i[0] * i[1] for i in zip(sent_freq1, sent_freq2))
        deno = sum(i ** 2 for i in sent_freq1) * sum(i ** 2 for i in sent_freq1)
        return (num / deno)


def rte_features(rtepair):
    extractor = RTEFeatureExtractor(rtepair)
    features = {}

    features['alwayson'] = True
    features['neg_txt'] = len(extractor.negwords & extractor.text_words)
    features['neg_hyp'] = len(extractor.negwords & extractor.hyp_words)

    features['word_overlap'] = len(extractor.overlap_new('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra_new('word'))
    features['ne_overlap'] = len(extractor.overlap_new('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra_new('ne'))
    #
    features['word_overlap_noun'] = len(extractor.overlap_new('noun'))
    features['word_overlap_verb'] = len(extractor.overlap_new('verb'))
    features['word_hyp_extra_noun'] = len(extractor.hyp_extra_new('noun'))
    features['word_hyp_extra_verb'] = len(extractor.hyp_extra_new('verb'))
    features['word_syn_overlap'] = extractor._syn_overlap_count
    features['cosine_similarity'] = extractor._cosine_sim
    # print(extractor._cosine_sim)

    # print(extractor._syn_overlap_count)


    # print("#####")
    # print(extractor.hyp_words)
    # print(extractor.text_words)
    # print(extractor.overlap_new('word'))
    # print(extractor.overlap_new('noun'))
    # print(extractor.overlap_new('verb'))
    # print("#####")

    return features


def rte_featurize(rte_pairs):
    return [(rte_features(pair), pair.value) for pair in rte_pairs]


def rte_classifier(algorithm, train_set , test_set):
    from nltk.corpus import rte as rte_corpus

    # train_set = rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])
    #test_set = rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])
    # test_set = rte_corpus.pairs(['COMP6751-RTE-10_TEST-SET_gold.xml'])
    # test_set = rte_corpus.pairs(['COMP6751-RTE-30_TEST-SET_gold.xml'])
    featurized_train_set = rte_featurize(train_set)
    featurized_test_set = rte_featurize(test_set)
    # Train the classifier
    print('Training classifier...')
    if algorithm in ['megam', 'BFGS']:  # MEGAM based algorithms.
        # Ensure that MEGAM is configured first.
        check_megam_config()
        clf = lambda x: MaxentClassifier.train(featurized_train_set, algorithm)
    elif algorithm in ['GIS', 'IIS']:  # Use default GIS/IIS MaxEnt algorithm
        clf = MaxentClassifier.train(featurized_train_set, algorithm)
    elif algorithm in ['NB']:  # Use default GIS/IIS MaxEnt algorithm
        clf = nltk.NaiveBayesClassifier.train(featurized_train_set)
    elif algorithm in ['DT']:  # Use default GIS/IIS MaxEnt algorithm
        # clf = nltk.DecisionTreeClassifier.train(featurized_train_set, binary=True, depth_cutoff=100, support_cutoff=20,entropy_cutoff=0.01)
        clf = nltk.DecisionTreeClassifier.train(featurized_train_set)

    else:
        err_msg = str(
            "RTEClassifier only supports these algorithms:\n "
            "'megam', 'BFGS', 'GIS', 'IIS'.\n"
        )
        raise Exception(err_msg)
    print('Testing classifier...')
    acc = accuracy(clf, featurized_test_set)
    print('Accuracy: %6.4f' % acc)
    return clf


def generateReport(test_set):
    print("creating performance report.........")
    featurized_test_set = rte_featurize(test_set)
    results = classifier.classify_many([fs for (fs, l) in featurized_test_set])
    y_true = [l for ((fs, l), r) in zip(featurized_test_set, results)]
    y_pred = classifier.classify_many([fs for (fs, l) in featurized_test_set])
    target_names = ['class 0', 'class 1']
    print(classification_report(y_true, y_pred, target_names=target_names))


def writeInxls(test_set , fileName):
    print("writing to xls...........")
    featurized_test_set = rte_featurize(test_set)
    results = classifier.classify_many([fs for (fs, l) in featurized_test_set])
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet Name")
    # Specifying style
    style = xlwt.easyxf('font: bold 1')
    i = 1
    sheet.write(0, 0, 'ID', style)
    sheet.write(0, 1,'Text', style)
    sheet.write(0, 2, 'Hyper', style)
    sheet.write(0, 3, 'alwayson', style)
    sheet.write(0, 4, 'word_overlap', style)
    sheet.write(0, 5, 'word_hyp_extra', style)
    sheet.write(0, 6, 'ne_overlap', style)
    sheet.write(0, 7, 'ne_hyp_extra', style)
    sheet.write(0, 8, 'word_overlap_noun', style)
    sheet.write(0, 9, 'word_overlap_verb', style)
    sheet.write(0, 10, 'word_hyp_extra_noun', style)
    sheet.write(0, 11, 'word_hyp_extra_verb', style)
    sheet.write(0, 12, 'neg_txt', style)
    sheet.write(0, 13, 'neg_hyp', style)
    sheet.write(0, 14, 'value', style)
    sheet.write(0, 15, 'predict', style)

    for x in test_set:
        features = rte_features(x)
        # print(str(j) + "-" + str(i))
        sheet.write(i, 0, str(x.id), style)
        sheet.write(i, 1, str(x.text), style)
        sheet.write(i, 2, str(x.hyp), style)
        sheet.write(i, 3, str(features['alwayson']), style)
        sheet.write(i, 4, str(features['word_overlap']), style)
        sheet.write(i, 5, str(features['word_hyp_extra']), style)
        sheet.write(i, 6, str(features['ne_overlap']), style)
        sheet.write(i, 7, str(features['ne_hyp_extra']), style)
        sheet.write(i, 8, str(features['word_overlap_noun']), style)
        sheet.write(i, 9, str(features['word_overlap_verb']), style)
        sheet.write(i, 10, str(features['word_hyp_extra_noun']), style)
        sheet.write(i, 11, str(features['word_hyp_extra_verb']), style)
        sheet.write(i, 12, str(features['neg_txt']), style)
        sheet.write(i, 13, str(features['neg_hyp']), style)
        sheet.write(i, 14, str(x.value), style)
        sheet.write(i, 15, str(results[i - 1]), style)
        i += 1
    workbook.save(fileName + ".xls")


if __name__ == '__main__':
    option = int(input("Enter 1: If you want to test RTE1 dataset \n"
                       "Enter 2:If you want to test RTE2 dataset \n"
                       "Enter 3: If you want to test RTE3 dataset \n"
                       "Enter 4:  if you want to test COMP6751-RTE-10_TEST-SET_gold.xml file \n"
                       "Enter 5:  If you want to test COMP6751-RTE-30_TEST-SET_gold.xml file \n"))
    if option == 1:
        # test_set = rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])
        test_set = rte_corpus.pairs(['rte1_test.xml'])
    elif option == 2:
        test_set = rte_corpus.pairs(['rte2_test.xml'])
    elif option == 3:
        test_set = rte_corpus.pairs(['rte3_test.xml'])
    elif option == 4:
        test_set = rte_corpus.pairs(['COMP6751-RTE-10_TEST-SET_gold.xml'])
    elif option == 5:
        test_set = rte_corpus.pairs(['COMP6751-RTE-30_TEST-SET_gold.xml'])
    else:
        fileName = (input("Enter test file Name \n"))
        test_set = rte_corpus.pairs([fileName])
    # Training the classifier using RTE1 , RTE2, RTE3 dev dataset
    train_set = rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])
    algorithm = (input("Enter Algoirthm , supported algorithms are 'GIS', 'IIS', 'DT' ,'NB' \n"))
    importReport = input(" Enter 'y', If you want to import the result in xls format, else press any key \n")

    if algorithm in ['GIS', 'IIS', 'DT','NB']:
        print("Classifier invoked ... ")
        classifier = rte_classifier(algorithm, train_set, test_set)
        generateReport(test_set)
    else:
        err_msg = str(
            "RTEClassifier only supports these algorithms:\n "
            "'GIS', 'IIS', 'DT' ,'NB'.\n")
        raise Exception(err_msg)
    if importReport == 'y':
        fileName = "default" if option == 1 else 'gold' + str(option)
        fileName = fileName + "_"+algorithm
        writeInxls(test_set, fileName)


