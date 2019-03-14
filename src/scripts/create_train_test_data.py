import sys
import pandas as pd
import numpy as np
import glob
import re

def find_word_bags(file_glob):
    rgx_singles = re.compile("([\w][\w']*[\w])")
    rgx_doubles = re.compile("([\w][\w']*[\w] +[\w][\w']*[\w])")
    rgx_triples = re.compile("([\w][\w']*[\w] +[\w][\w']*[\w] +[\w][\w']*[\w])")


    words = []
    for filename in file_glob:
        file = open(filename, 'r')
        for line in file:
            words += rgx_singles.findall(line)
            words += rgx_doubles.findall(line)
            words += rgx_triples.findall(line)

    return words


def create_feature_matrix(file_glob, dictionary):
    rgx_singles = re.compile("([\w][\w']*[\w])")
    rgx_doubles = re.compile("([\w][\w']*[\w] +[\w][\w']*[\w])")
    rgx_triples = re.compile("([\w][\w']*[\w] +[\w][\w']*[\w] +[\w][\w']*[\w])")

    matrix = []
    for filename in file_glob:
        encoded_words = np.zeros(len(word_features))

        file = open(filename, 'r')
        for line in file:
            words = rgx_singles.findall(line)
            words += rgx_doubles.findall(line)
            words += rgx_triples.findall(line)
            for word in words:
                if (word in dictionary):
                    encoded_words[dictionary[word]] += 1

        matrix += [encoded_words]
    return matrix


if __name__ == '__main__':
    # Get the total number of args passed to the prediction.py
    total_args = len(sys.argv)
    if total_args == 5:
        # Get the arguments list
        cmdargs = sys.argv
        n_words = int(cmdargs[1])
        train_dir = cmdargs[2]
        test_dir = cmdargs[3]
        out_dir = cmdargs[4]


    else:
        print('ERROR: incorrect arguments!')
        print('create_train_test_data.py <n_words> <train_dir> <test_dir> <output_dir> ')
        sys.exit(-1)

    print('Creating dictionary...')
    pos_words = find_word_bags(glob.glob(train_dir + '/pos/*.txt'))
    neg_words = find_word_bags(glob.glob(train_dir + '/neg/*.txt'))

    df_words = pd.DataFrame(pos_words + neg_words, columns=['word'])
    df_words['count'] = 1

    word_count = df_words.groupby('word').count()
    word_features = list(word_count.sort_values('count', ascending=False).head(n_words).index)

    dictionary = {}
    for idx in range(len(word_features)):
        dictionary[word_features[idx]] = idx

    print('done.')

    #
    # Now that the dictionary is set. We build the feature matrices
    #

    # Train
    print('Preparing training data...')
    pos_matrix = create_feature_matrix(glob.glob(train_dir + '/pos/*.txt'), dictionary)
    neg_matrix = create_feature_matrix(glob.glob(train_dir + '/neg/*.txt'), dictionary)

    x_pos_train = pd.DataFrame(pos_matrix, columns=word_features)
    x_pos_train['target'] = 1

    x_neg_train = pd.DataFrame(neg_matrix, columns=word_features)
    x_neg_train['target'] = 0

    df_train = pd.concat([x_pos_train.drop('br', axis=1), x_neg_train.drop('br', axis=1)])
    df_train.to_csv(out_dir + '/train__' + str(n_words) + '.csv', index=False)

    # Test
    print('Preparing test data...')
    pos_matrix = create_feature_matrix(glob.glob(test_dir + '/pos/*.txt'), dictionary)
    neg_matrix = create_feature_matrix(glob.glob(test_dir + '/neg/*.txt'), dictionary)

    x_pos_test = pd.DataFrame(pos_matrix, columns=word_features)
    x_pos_test['target'] = 1

    x_neg_test = pd.DataFrame(neg_matrix, columns=word_features)
    x_neg_test['target'] = 0

    df_test = pd.concat([x_pos_test.drop('br', axis=1), x_neg_test.drop('br', axis=1)])
    df_test.to_csv(out_dir + '/test__' + str(n_words) + '.csv', index=False)

    print('done.')


