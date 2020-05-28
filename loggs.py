import os

class Loggs(object):
    def __init__(self, params, filename='loggs.txt'):
        self.params = params
        self.filename = filename
        with open(filename, 'w') as f:
            f.write(','.join(list(params))+'\n')

    def save(self, params):
        with open(self.filename, 'a') as f:
            f.write(','.join(list(map(str, params)))+'\n')


def training_bar (iter, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    filledLength = int(length * iter // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s/%s | %s' % (prefix, bar, iter, total, suffix), end = printEnd)
    # Print New Line on Complete
    if iter == total: 
        print()
