import pandas as pd

from data_graph_RNA import create_ppi_graph

pd.set_option('display.max_colwidth', None)

case_id = 'fd5c44ef-ea50-4fba-9e8d-e371cf34ebdb' # for example

ppi_data = create_ppi_graph(case_id)
