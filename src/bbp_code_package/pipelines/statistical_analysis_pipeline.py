from kedro.pipeline import Pipeline, node, pipeline

import bbp_code_package.nodes.statistical_analysis.statistical_analysis_main as stat_analysis


def statistical_analysis(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=stat_analysis.run_statistical_analysis_1_v_all,
                inputs=["cells_reformated", "parameters"],
                outputs=[
                    "statistical_significance_1vall",
                    "statistical_analysis_1vall",
                ],
                name="stat_analysis_1vall",
                tags="stat_analysis_1vall",
            ),
            node(
                func=stat_analysis.run_statistical_analysis_1_v_1,
                inputs=["cells_reformated", "parameters"],
                outputs=[
                    "stat_analysis_1v1",
                    "stat_synthesis_1v1",
                ],  # ["statistical_significance", "statistical_analysis"],
                name="stat_analysis_1v1",
                tags="stat_analysis_1v1",
            ),
        ]
    )
