import os


# Dummy kernels are benchmark scaffolding and stay out of the default
# correctness suite unless explicitly opted in.
collect_ignore = []
if os.environ.get('BRAINEVENT_INCLUDE_DUMMY_PYTEST') != '1':
    collect_ignore.append('dummy_backend_test.py')
