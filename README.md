# Typologically Diverse Language Sampling

Code for the paper etc. Note that this readme (and the repository in general) are a work in progress.

## Usage
### Installation
Normal:
```
pip install git+https://github.com/esther2000/typdiv-sampling.git
```

With visualization support:
```
pip install "git+https://github.com/esther2000/typdiv-sampling.git[vis]"
```

*instructions on data prep script here*

### Example
```python
from typdiv_sampling import Sampler

# A list of glottocodes to sample from.
frame = ["haio1238", "mehi1240", "nina1238", "duun1241", "turk1308", "nyan1301"]
k = 3  # The number of languages to sample.
seed =  # A random seed for the non-deterministic methods.

# Initialize with default setup.
sampler = Sampler(
    dist_path=dist_path, # Language distances to use.
    gb_path=gb_path, # Grambank csv file.
    wals_path=wals_path, # WALS csv file
    counts_path=counts_path, # Language count file.
)
sampler.sample_mdp(frame, k)
> ['nucl1347', 'tami1289', 'thai1261'] 
```

---

### Data

Here are two steps to do before language sampling:
*Note: I'll make this nicer later.*

1. Get Grambank (note to self: cite)
2. Calculate language distances: this can be done with `data/compute_all_distances.py` (note to self: file too big for github)


### Terminology
- `k`: the number of languages to sample (also: `max_langs`)
- `N`: languages to sample from (also: `frame`)

### Language sampling

You can sample k languages from sampling frame N as follows:

`python main.py -s <SAMPLING_METHODS> -k <NUM_LANGS> -f <ALL_LANGS> -d <LANG_SIM>`

Below is an example, where we do MDP sampling to select 10 languages from all in Grambank, using default language distance calculation:

`python main.py -s typ_mdp -k 10 -f data/frames/langs_gb.txt -d data/gb_vec_sim_0.csv`

The resulting samples are written to `evaluation/samples/`, where the filename corresponds to the arguments.


### Evaluation

Evaluation (currently: entropy only), given a language sample, can be done as follows:

`python evaluation/typdiv_eval.py -s <SAMPLE>`, where the sample is the output of `main.py`
