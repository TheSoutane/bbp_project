from kedro.pipeline import Pipeline, node, pipeline

import bbp_code_package.nodes.extraction.cell_data_load_process as load
import bbp_code_package.nodes.extraction.reporting_utils as report


def extract_data(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load.load_preprocess_mat_file,
                inputs=["CellList", "parameters"],
                outputs="cells_extracted_raw",
                name="mat_file_formatting",
                tags="raw_extraction",
            ),
            node(
                func=report.get_report_02_intermediate,
                inputs=["cells_extracted_raw", "parameters"],
                outputs="outliers_df_02",
                name="report_02",
                tags="report_02",
            ),
        ]
    )
