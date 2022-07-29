from kedro.pipeline import Pipeline, node, pipeline

import bbp_code_package.nodes.extraction.reporting_utils as report
import bbp_code_package.nodes.reformatting.reformatting_main as reformating


def reformat_data(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=reformating.reformat_dataframe,
                inputs=["cells_extracted_raw", "parameters"],
                outputs="cells_reformated",
                name="raw_file_reformatting",
                tags="reformatting",
            ),
            node(
                func=report.get_report_03_primary,
                inputs=["cells_reformated", "parameters"],
                outputs="outliers_df_03",
                name="report_03",
                tags="report_03",
            ),
        ]
    )
