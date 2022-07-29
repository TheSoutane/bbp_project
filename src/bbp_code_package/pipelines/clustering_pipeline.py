from kedro.pipeline import Pipeline, node, pipeline

import bbp_code_package.nodes.clustering.clustering_main as clustering


def run_clustering(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clustering.prepare_clustering_input,
                inputs=["cells_reformated", "parameters"],
                outputs="clustering_input",
                name="clustering_input",
                tags="clustering_input",
            ),
            node(
                func=clustering.run_clustering,
                inputs=["clustering_input", "parameters"],
                outputs=[
                    "clustering_output",
                    "centroids",
                ],
                name="clustering_calc",
                tags="clustering_calc",
            ),
        ]
    )
