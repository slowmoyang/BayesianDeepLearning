import numpy as np

from uncertainty import compute_variation_ratio
from uncertainty import compute_predictive_entropy
from uncertainty import compute_mutual_information

def get_examples(num=1000):
    assert num % 2 == 0

    y1 = [np.ones(num, dtype=np.float32), np.zeros(num, dtype=np.float32)]
    y1 = np.stack(y1)
    y1 = y1.transpose(1, 0)
    y1 = y1.reshape(num, 1, 2)

    y2 = 0.5 * np.ones((num, 1, 2))

    half = int(num / 2)
    y3 = [np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32)]
    y3 = np.stack(y3)
    y3 = y3.transpose(1, 0)
    y3 = np.concatenate([y3, y3[:, ::-1]]).reshape(num, 1, 2).astype(np.float32) 
    return y1, y2, y3


def _test():
    y1, y2, y3 = get_examples()

    solns = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0.5, 0.5]
    ]

    for i, (s, y) in enumerate(zip(solns, [y1, y2, y3])):
        print("{:d}: {} <--> {}".format(i, s[0], compute_variation_ratio(y)))
        print("{:d}: {} <--> {}".format(i, s[1], compute_predictive_entropy(y)))
        print("{:d}: {} <--> {}".format(i, s[2], compute_mutual_information(y)))
        print("@" * 30)

    print("log(1/2) == {}".format(np.log(0.5)))


def main():
    _test()


if __name__ == '__main__':
    main()
