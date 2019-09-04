import random
from collections import Counter

class Markov_Model(object):

    def __init__(self, max_length=1, is_most=False):
        self.markov_table = {}
        self.max_length = 1
        self.is_most = False

    def create_model(self, token_list, max_depth=1, is_most=False):
        '''
        create a markov model with the depth from 1 to max_depth
        {
            depth1:{
                key1:[value1, value2 ..]
            }
        }
        '''
        self.is_most = is_most
        self.max_length = max_depth
        for depth in range(1, max_depth+1):
            temp_table = {}
            for index in range(depth, len(token_list)):
                words = tuple(token_list[index-depth:index])
                if words in temp_table.keys():
                    temp_table[words].append(token_list[index])
                else:
                    temp_table.setdefault(words, []).append(token_list[index])
            if is_most:
                for key,value in temp_table.items():
                    temp = Counter(value).most_common(1)[0][0]
                    temp_table[key] = temp
                self.markov_table[depth] = temp_table
            else:
                self.markov_table[depth] = temp_table
        return self.markov_table



    def query_test(self, pre_tokens, depth=1):
        while(depth>self.max_length):
            depth -= 1

        used_tokens = pre_tokens[len(pre_tokens)-depth:]
        used_tokens = tuple(used_tokens)
        while used_tokens not in self.markov_table[depth].keys() and depth > 1:
            depth -= 1
            used_tokens = tuple(pre_tokens[-depth:])
        if self.is_most:
            candidate = self.markov_table[depth][used_tokens]
        else:
            candidate_list = self.markov_table[depth][used_tokens]
            random_index = random.randint(0, len(candidate_list)-1)
            candidate = candidate_list[random_index]
        return candidate



if __name__ == '__main__':
    markov_model = Markov_Model()
    string_token_list = ['to','be', 'or', 'not', 'to', 'be', 'that', 'is', 'a', 'question']
    markov_table = markov_model.create_model(string_token_list, max_depth=3, is_most=False)
    print(markov_table)
    # test_string = ['a','to','be']
    # prediction = markov_model.query_test(test_string)
    # print(prediction)
