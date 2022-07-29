"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*reform_utils.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pandas as pd
import pytest
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings

import tests.pipelines.conftest as conftest
from bbp_code_package.nodes.extraction.mat_file_extraction import (
    extract_preformat_mat_file,
)


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="bbp_code_package",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality


class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

    def test_data_formatting(self, project_context):
        params = project_context.params
        test_cell_id, test_cell_dict = conftest.return_processed_cell_202_1()
        tmp, apwaveform_to_test, idrest_to_test = extract_preformat_mat_file(
            f"aCell{test_cell_id}", params
        )

        df_to_test = pd.concat([apwaveform_to_test, idrest_to_test], axis=1)

        for feature in test_cell_dict.keys():
            print(feature, type(test_cell_dict[feature]), type(df_to_test[feature]))
            assert round(test_cell_dict[feature], 5) == round(df_to_test[feature][0], 5)
