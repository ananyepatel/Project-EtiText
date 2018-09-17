from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

import os
from fasttext import supervised


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


train_data = os.path.join(os.getenv("DATADIR", ''), 'codingpreprocessed.train')
valid_data = os.path.join(os.getenv("DATADIR", ''), 'codingpreprocessed.valid')

# bi-grams show best performance
model = supervised(input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1)
print_results(*model.test(valid_data))

model = supervised(input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1, loss="hs")

print_results(*model.test(valid_data))
model.save_model("coding.bin")
