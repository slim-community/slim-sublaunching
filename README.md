This repository is for code/scripts that help with _sublaunching_: running a given SLiM model more than once...

- either for replication or for exploring different parameter values, and...

- either on the local machine or on a computing cluster of some kind.

At present, this repo contains:

**basic_r_usage_aggregateLocalAncestry.R** : an R script to replicate recipe 13.9 1000 times, aggregate the results, and produce a plot.

**basic_python_usage_gen_replicates.py** : a Python script to perform replicated runs and tabulate a binary metric from their output.

**sublaunching_tutorial** : a tutorial on sublaunching via BASH, Python, or R that demonstrates a full workflow for running an included demo SLiM file through a grid of parameters.

This is not a single-author repository!  These contributions are all by different people (see the credits on each item), and we'd welcome new pull requests that show other approaches to the problem of sublaunching.  It's a problem that pretty much every SLiM user has to figure out at some point, it's a frequently asked question at SLiM workshops and on slim-discuss, and there are many different ways to handle it!
