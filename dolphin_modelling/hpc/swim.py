# -*- coding: utf-8 -*-

import sys
from dolphin.processor import Processor


processor = Processor('../')

lens_name = str(sys.argv[1])
model_id = str(sys.argv[2])


processor.swim(lens_name, model_id=model_id, recipe_name='galaxy-galaxy',mpi=True)