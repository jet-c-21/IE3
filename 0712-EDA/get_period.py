from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pprint import pprint as pp
from tqdm import tqdm


def get_nrlz_col(val):
    val = val.values.reshape(-1, 1)
    # scaler = preprocessing.StandardScaler().fit(val)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(val)
    return scaler.transform(val)


COMMENTS_PATH = 'IBI_COMMENTS_TRAIN.csv'
COMMENTS = pd.read_csv(COMMENTS_PATH, low_memory=False)

DICTIONARY_PATH = 'challenge_dictionary_info.csv'
DICTIONARY = pd.read_csv(DICTIONARY_PATH, low_memory=False)
DICTIONARY = DICTIONARY.sort_values(by=['count'], ascending=False)

ALL_PUNCT = DICTIONARY[(DICTIONARY['pos'] == 'PUNCT') & (DICTIONARY['count'] > 1)].reset_index(drop=True)
ALL_PUNCT['count_std'] = get_nrlz_col(ALL_PUNCT['count'])
CANDIDATE = ALL_PUNCT[ALL_PUNCT['count_std'] > 0.12]


def get_com_dict():
    result = dict()
    result['cf'] = 0
    result['last_count'] = 0
    result['grammar'] = 0

    return result


def is_at_last(word: str, comment: str) -> bool:
    tokens = comment.split(' ')
    if tokens[-1] == word:
        return True
    else:
        return False


def is_valid_sent(sentence, dictionary):
    result = False
    sentence = sentence.strip()
    words = sentence.split(' ')

    has_noun = False
    for w in words:
        pos = dictionary[dictionary['id'] == w]['pos'].values
        if pos == 'PRON':
            has_noun = True

        if has_noun and pos == 'VERB':
            result = True
            break

    return int(result)


def get_grammar_score(word: str, comment: str, dictionary: pd.DataFrame):
    sentences = comment.split(word)
    check = [is_valid_sent(sent, dictionary) for sent in sentences]
    return np.mean(check)


def parse_comment(data):
    result = get_com_dict()
    word = data['word']
    comment = data['comment']
    dictionary = data['dictionary']

    if word in comment:
        result['cf'] += 1

        if is_at_last(word, comment):
            result['last_count'] += 1

        # grammar_score = get_grammar_score(word, comment, dictionary)
        # result['grammar'] += grammar_score

        print('done')
        return result

    else:
        print('done')
        return result


if __name__ == '__main__':
    FINAL = pd.DataFrame(columns=['id', 'cf', 'cfr', 'last_count', 'last_rate', 'grammar', 'grammar_rate'])

    for cand in CANDIDATE['id']:
        cf = 0
        cfr = 0

        last_count = 0
        last_rate = 0

        grammar = 0
        gr = 0

        print('do: {}'.format(cand))

        with Pool(4) as p:
            res = tqdm(p.map(parse_comment, [{'word': cand, 'comment': d.strip(), 'dictionary': DICTIONARY}
                                                        for d in COMMENTS['NOTES']]), total=len(COMMENTS['NOTES']))

        for record in res:
            cf += record['cf']
            last_count += record['last_count']
            grammar += record['grammar']

        cfr = cf / len(COMMENTS)
        last_rate = last_count / len(COMMENTS)
        gr = grammar * cfr

        FINAL.loc[len(FINAL)] = [cand, cf, cfr, last_count, last_rate, grammar, gr]

    FINAL.to_csv('PERIOD_CAND.csv', encoding='utf-8', index=False)
