import platform
import re
import warnings
import os
import uuid
import multiprocessing

# filename = '{}/{}.yml'.format(os.path.dirname(os.path.abspath(__file__)), basename)

# with open(filename, 'r') as f:
# 	variables = yaml.load(f)

# if not variables:
# 	warnings.warn('No variables in {}.'.format(filename))

# unused_vars = set(variables.keys()).difference(set(globals().keys()))
# if len(unused_vars) > 0:
# 	warnings.warn('The folllowing variables defined are not used:')
# 	for v in unused_vars: print(v)

# globals().update(variables)


################################### Initialize 
UNIQUE_ID = None
MODEL_SAVING_INTERVAL = 7200
EVALUATION_INTERVAL = 120
EPSILON = 1e-8

################################### Obtain unique ID:
# Define regular expression
__config_file_mapping = {
	r'(flux-login\d+|nyx\d+)\.arc-ts\.umich\.edu' : 'flux',
	r'(compute-\d+\.)?vl-fb\.eecs.umich\.edu' : 'vl-fb',
	r'v5' : 'v5',
	r'compute-10' : 'vl-fb'
}
# Load corresponding configure file using hostname
hostname = platform.node()
basename = next(bname for regex, bname in __config_file_mapping.items() if re.match(regex, hostname))
BASENAME = basename
print('Detected platform: {}'.format(basename))

if basename == 'vl-fb': # Slurm
	UNIQUE_ID = os.environ.get('SLURM_JOB_NAME')
	if UNIQUE_ID is None:
		UNIQUE_ID = 'debug'
elif basename =='flux': # PBS
	UNIQUE_ID = os.environ.get('PBS_JOBNAME')
	if UNIQUE_ID is None:
		UNIQUE_ID = 'debug'	
elif basename == 'v5':
	UNIQUE_ID = 'debug'
else:
	raise NotImplementedError


JOBS_MODEL_DIR = "./exp/%s/models" % UNIQUE_ID
JOBS_LOG_DIR = "./exp/%s/log" % UNIQUE_ID
JOBS_DIR = './exp/%s' % UNIQUE_ID
print "Unique_ID: {}".format(UNIQUE_ID)
print "Job folder:{}".format(JOBS_DIR)

################################### Get number of cores
NUM_CORES = multiprocessing.cpu_count()
print "NUM_CORES = ", NUM_CORES
