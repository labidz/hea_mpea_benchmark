import json, sys
def nb_from_cells(cells, path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, src in enumerate(cells):
        nb["cells"].append({"cell_type": "code", "execution_count": None, "id": f"c{i}", "metadata": {}, "outputs": [], "source": src})
    json.dump(nb, open(path, "w"), indent=1)
