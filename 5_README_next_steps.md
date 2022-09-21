# bbp_electrophysiology_analysis

## Next steps

Here is a list of the potential next actions. They are structured in 3 main points

* MLOps
  * Solidify the unit tests and extend it
    * To more protocls
    * To more coding steps
  * Restructure the parameters dict in a more solid structure:
    * Node dict ?
  * integrate the visualisation tools in a more robust software
    * Deploy app on a server and make it easily accessible
* Data integration
  * Integrate more cells (to be discussed with Liviu/Rajnish)
  * Integrate more protocols (few are missing)
  * Integrate more repetitions
* Research solidification
  * Simplification of the data
    * The current dataframe curently has a lot of features (1k<). It is therefore necessary to simplify it. Multiple options are to be considered
      * PCA: Run priincipal component analysis and simplify data accordingly
      * Co-correlation simplification: Study correlation between features and simplify data by keeping one from a highly correlated group. This approach can be combined with PCA
      * Iterate on cluster analysis to identify the main segregation features and go further with a subsample.
  * Data augmentation
    * Find links and transformation between data to potentially improve the data set size. A structure like Unet might be required

I recommend to work on the simplification point, with a combination of the 3 approaches