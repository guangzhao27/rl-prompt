import random
import sys
import os
import numpy

sys.path.append("/pscratch/sd/g/gzhao27/rl-prompt/InstOptima")

from entity.population import Population

# first finish coding instruction and instruction operation
from entity.instruction import Instruction
p = Instruction(dataset="yelp")

from operators.instruction_operators import InstructOperator
inst_op = InstructOperator()

inst_op.evolve(p, p)

# second finish coding individual and objective

#thrid finish coding Population

print('end')