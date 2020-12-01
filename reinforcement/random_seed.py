from numpy.random import Generator, MT19937, SeedSequence
import sys
sg = SeedSequence(1234)
print("random seeds ",MT19937(sg).random_raw(int(sys.argv[1])))
