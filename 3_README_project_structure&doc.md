# bbp_electrophysiology_analysis

## The KEDRO framework

This project is structured using the Kedro framework. In order to understand how to use it I highly recommend to read the [docs](https://kedro.readthedocs.io/en/stable/index.html) and to do the ["spacecafts tutorial"](https://kedro.readthedocs.io/en/stable/tutorial/spaceflights_tutorial.html)
This framework structures the code and centralise actions to ensure best practices. On top of the structuring and orchestration, It facilitates the use of unit tests, code linting and other good practices.


## Nodes and Pipelines

Kedro structures the code into pipelines and nodes that can be run independently. 
A pipeline is a succession of nodes and a noe is a succession of functions transforming an input into an output.

### Nodes

The detailed documentation is available [here](https://kedro.readthedocs.io/en/latest/nodes_and_pipelines/nodes.html#nodes)

Nodes are the building blocks of a kedro project. They are defined by input(s) and output(s). They can be seen as function orchestrators.
The underlying code is stored in the `src/bbp_code_package/nodes` folder. Each subfolder contains the code specific to one node.
The nodes are defined in the `src/bbp_code_package/pipelines` folder (I know, this is counter intuitive...)
Each node is defined with its input(s), output(s) and corresponding function. The function is defined in the nodes folder and the input/output are defined in the data catalog (see next section)
The sequence of nodes will be the base of the pipeline. They can be run individually with the following command:
```
kedro run --node=node_name
```

### Pipelines

The detailed documentation is available [here](https://kedro.readthedocs.io/en/latest/nodes_and_pipelines/pipeline_introduction.html)

A pipeline is an orchestrator of the main project steps. For example, the data extraction, formatting, modeling, ...
They are registered in the `src/bbp_code_package/pipeline_registry.py` file and defined in the `src/bbp_code_package/pipelines` file.
They can also be ran individually using the following command
```
kedro run --pipeline=pipeline_name
```
### List of pipelines and nodes

Here is the list of pipelines and corresponding nodes with a brief description

* Data extraction:
  * **Descr**: Sequentially loads the raw .mat files, format it in a tabular form and perform basic statistical and quality analysis
  * **Nodes**:
    * load_preprocess_mat_file: Loads and process raw input. The .mat files are imported into python and their format is changed from a nested structure to a tabular structure
      * _inputs_:
        * CellList: List of cells to be processed
        * Parameters: Dict of code parameters
      * _outputs_: 
        * cells_extracted_raw: Dataframe with one row per cell containing multiple features for the different protocols
    * get_report_02_intermediate: Produce a pdf report to analyse visually the data quality and distribution
        * !! This node was commented as it takes a lot of time to create the plots
        * _inputs_:
          * cells_extracted_raw: Output of the loading Node
          * Parameters: Dict of code parameters
        * _outputs_: 
          * outliers_df_02: Dataframe containing outliers (defined with a basic threshold). Those features need to be investigated
          * (not in the data catalog) Plots for each features and an aggregated pdf. There is one plot per feature, showing its distribution, and basic statistics
* Data formatting: 
  * **Descr**: Process the raw extracted data to prepare it for analysis and produces a second report
  * **Nodes**:
    * reformat_dataframe:
      * _inputs_:
        * cells_extracted_raw: Output of the loading Node
        * Parameters: Dict of code parameters
      * _outputs_: 
        * cells_reformated: Cleaned dataframe where outliers have been removed/capped
    * get_report_03_primary
      * !! This node was commented as it takes a lot of time to create the plots
        * _inputs_:
          * cells_reformated: Output of the Reformating Node
        * _outputs_: 
          * outliers_df_03: Dataframe containing outliers (defined with a basic threshold). Those features need to be investigated
          * (not in the data catalog) Plots for each features and an aggregated pdf. There is one plot per feature, showing its distribution, and basic statistics
* Statistical analysis:
  * **Descr**: Performs statistical analysis over the dataframe 
  * **Nodes**:
    * run_statistical_analysis_1_v_all
      * _inputs_:
        * cells_reformated: Output of the Reformating Node
        * Parameters: Dict of code parameters
      * _outputs_: 
        * statistical_significance_1vall: List of statistical significanes of each celltype VS all the other cells for each feature
        * statistical_analysis_1vall: Synthesis of the previos file
    * run_statistical_analysis_1_v_1
      * _inputs_:
        * cells_reformated: Output of the Reformating Node
        * Parameters: Dict of code parameters
      * _outputs_: 
        * stat_analysis_1v1: Satistical significance test performed for each celltype_1/celltyp_2/feature combination
        * stat_synthesis_1v1:
* Clustering:
  * **Descr**: Prepare data for clustering and runs analysis
  * **Nodes**:
    * prepare_clustering_input
      * _inputs_:
        * cells_reformated: Output of the Reformating Node
        * Parameters: Dict of code parameters
      * _outputs_: 
        * clustering_input: Data prepared for clustering (with added features, one hot encoding, ...)
    * run_clustering
      * _inputs_:
        * cells_reformated: Output of the Reformating Node
        * Parameters: Dict of code parameters
      * _outputs_: 
        * clustering_output: Input data with new columns attibution clusters to the cells
        * clustering_output_unsc: De scaled clustering output
        * centroids: Description of the clustering centroids

## Other elements

In addition to the pipelines, multiple elements are present and usablefor this project

### Configuration files

Defined by the Kedro framework, three configuration files are located in the `conf/base` folder

* catalog.yml
  * This is the [kedro datacatalog](https://kedro.readthedocs.io/en/latest/data/data_catalog.html?highlight=catalog)
  * This file stores the path and nature of all the data used. It needs to be configured before the creation of nodes (ex, the name of an output in a node refers to the corresponding datacatalog element)
* parameters.yml
  * This big .yml file stores all the project parameters in one dict. Is is separated in subdicts, specific to nodes and/or protocols
* viz_parameters.yml
  * Small parameters dict used for the visualization tools

### Visualization tools

The `src/bbp_code_package/visualisation_app` folder contains multiple sripts running dash dashboards. To run them, please go to the project folder (ecode_analysis) and run the following script:
```
python src/bbp_code_package/visualisation_app/dash_script_name.py
```
and open the link displayed in the terminal.

Here are the relevant scripts (the others are templates to be used for further development):
* box_scatter_viz.py
  * Allows to display box plots or scatterplots of features. The first box defines the plot type, the second the grouping type (celltype, species, ...) and the 2 last ones the features to be explored
* cluster_analysis.py
  * Tools to quickly analyse cluster. It display clustr statistics and average value for the selected features. Values are automatically colored in blus if they significantly differs from the average of cells

### Sphinx .html documentation

Sphinx is a framework that automates the building of documentation. Nevertheless I had issues making it working. 
The docs are produced but the toctree (link between the .html files) has issues. If you want a deep dive in the implemented functions, please have a look ar the following files in the 
`docs/build/html/` folder:
* bbp_code_package.nodes.clustering
* bbp_code_package.nodes.extraction
* bbp_code_package.nodes.reformating
* bbp_code_package.nodes.statistical_analysis
* bbp_code_package.pipelines

### Unit tests

Some unit tests have been implemented at the beguinning of the prodject. Sadly I hadn't the opportunity to update/complet those. 
Feel free to update/extend them. They are located in the `src/tests/pipelines` folder.

### Notebooks

Located in the `notebooks` file, multiple notebooks have been developped to explore data and develop solutions. 

Here is the list:

* Cell_features_exploration.ipynb: Illustrative flattening of a cell
* dataframe comparison.ipynb: Comparison of 2 dataframes
* function test.ipynb: Framework to load and run a function from the kedro script
* Illustrative flattening_APWaveform.ipynb: Illustrative flattening of an APW cell (General method)
* Illustrative flattening_NegCheops.ipynb: Illustrative flattening of an APW cell (Different structure)
* raw_data_exploration.ipynb: framework to explore the raw extraction results
* test_notebooks
  * Loading function test.ipynb: Compares the results of the script loading with a manual one to ensure variables consistency
* function development: Those notebooks were used to develop some of the code functions. Feel free to explore them for a better understanding of further developments. 
  * clustering_implementation.ipynb:
  * clustering_viz_dev.ipynb:
  * mann_whitney_test.ipynb:
  * manova_dev.ipynb:
  * Reporting test.ipynb:
  * statistical analysis.ipynb:
  * statistical_test_analysis.ipynb:
  * stat_synth_reformatting.ipynb:
  * t_test_dev.ipynb


