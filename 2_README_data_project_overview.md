# bbp_electrophysiology_analysis

## Project Overview

This project to structures existing data and to perform analysis. The goal is to consolidate the existing 
discoveries and intuitions of the BBP using a robust statistical analysis. In a second time prospective analysis might also be performed. 
As you can guess, this is the final step of a broader pipeline. The data went through multiple steps before bbeing processed in this project. 
First of all, they are the result of a biological experiment performed on mouses and rats. In a second time, they are cleaned and structured in a much larger database before being filtered and provided to the efeatures project.

## Data extraction and experiments

The data we work on are derived from elecroancephalograms preformed on rat/mouses brain. the experimental steps are the following ones:
* The mouse is killed and its brain is extracted
* The brain is cut in tiny slices
* The slices corresponding to the selected brain areas (cortex, amygdala, ...) are kept while the others are discarded
* The cells are stimulated with one electrode and the output is measured with a second one
  * The connection of the electrodes are made using the [patch-clamp protocol](https://en.wikipedia.org/wiki/Patch_clamp)
    * Feel free to ask the lab members about the protocol and to see an experiment
  * Multiple stimulation protocol are applied (APWaveform, IDrest, IV, ...) with different intensity
    * The protocol is the "shape" of the stimulation (one/multiple steps, ramp, sinusoid,...)
    * For each protocol, multiple intensities are applied (ex: 100, 150, 200,... pV)
  * If possible, the experiment is replicated on the same cell
    * 2-3 times in general
* The output data (time series) are saved and stored


## Post processing and structure

* Post processing
  * The raw output of each stimulation is a time series, that is difficult to exploit directly. Therefore multiple features are extracted so it can be analysed
  * For each trace (time series) features are extracted. We can identify 3 types of features
    * Repetition features, that are linked to the cell and constant across the stimulations
    * Trace features, that are linked to a specific trace (ex: vHold, electric potential of the neuron before stimulation)
    * Spike features, linked to a spike. The number can vary from one cell to the other, even if the stimulation is the same.
  * The multiple experiments as well as the cell data are stored in a nested structure in matlab files (.mat). The following tree is an illustrative example

```
aCell_file
│
└───Cell data
│   │   Cell_Id
│   │   Celltype
│   │   experimenter
│   │   ...
│   
└───Protocol data
│    └───protocol_1
│    │    └───repetion_1
│    │    │   feature_1
│    │    │   feature_2
│    │    │   ...
│    │    └───repetion_2
│    │    │   feature_1
│    │    │   feature_2
│    │    │   ...
│    │    └───...
│    └───protocol_2
│    │    └───repetion_1
│    │    │   feature_1
│    │    │   feature_2
│    │    │   ...
│    │    └───repetion_2
│    │    │   feature_1
│    │    │   feature_2
│    │    │   ...
│    │    └───...
│    └───protocol_3
│    │    └───...
....
```

* Storage and cleaning
  * The data are stored in a much larger database
  * Clean data from this database are extracted and processed for our use
  * To know more about it, please contact Liviu Popescu


## Project Goal

As mentioned previously, the goal of this project is to use a quantitative approach to solidify the existing discoveries and/or intuitions
Here is a large list of what have been implemented and the potential next steps:

* Implemented
  * Processing
    * Data processing pipeline transforming the nested data into a tabular format
  * Analysis
    * Data analysis tools showing the distribution of the data and proposing cutoffs and cleaning thresholds
    * Data visualisation tools (.html files) allowing an interactive exploration of the data
  * Clustering
    * Clustering pipeline structure
    * Interactive analysis tool to speed up the iteration on clustering
* Next steps
  * Processing
    * Cleaning of the data based on human approach
      * Identify outliers and correct the corresponding cell
    * Solidification of the process (code improvement)
  * Analysis
    * Human analysis to extract/confirm conclusions
    * Simplification of the analysis but removing/aggregating features
      * Knowledge approach: Aggregate based on expert knowledge
      * Statistical approach: 
        * PCA (not recommended as we loose explainability)
        * Clustering: Iterative approach that can easily be combine with insights from the PCA and expert knowledge
    *Development of the clustering approach 