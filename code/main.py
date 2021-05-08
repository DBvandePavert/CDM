"""
CDM Research project by
- Daniel van de Pavert
-
-
-

something something main description
"""

from dataloader import *
from preprocess import *
from models import *


def main(input):
    load_some_data()
    foo = make_foo(input)
    bar = learn_something(foo)
    print(bar)


if __name__ == '__main__':
    main('PyCharm')
