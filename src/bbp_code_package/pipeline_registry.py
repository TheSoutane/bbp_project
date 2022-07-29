"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

import bbp_code_package.pipelines.data_extraction_pipeline as data_extr
import bbp_code_package.pipelines.data_formatting_pipeline as data_form
import bbp_code_package.pipelines.statistical_analysis_pipeline as stat_analysis
import bbp_code_package.pipelines.clustering_pipeline as clustering


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "__default__": data_extr.extract_data()
        + data_form.reformat_data()
        + stat_analysis.statistical_analysis(),
        "data_extraction": data_extr.extract_data(),
        "data_formatting": data_form.reformat_data(),
        "stat_analysis": stat_analysis.statistical_analysis(),
        "clustering": clustering.run_clustering(),
    }
