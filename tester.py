from utils import get_contraction_tree_from_graph, print_contraction_tree
import verifier as ver
from pathlib import Path
import typer
import os
from logger import logger, logging
import twin_width.parser as parser
import networkx as nx

app = typer.Typer()
BASE_PATH = Path(__file__).parent


def from_dir(instance_path: Path):
    instance_path = (BASE_PATH / instance_path).resolve()
    if not os.path.exists(instance_path):
        logger.error(f"folder is not exists {instance_path}")
    files = sorted(os.listdir(instance_path))
    files = filter(lambda file_name: file_name.endswith(".gr"), files)
    return files


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


@app.command()
def hello():
    print('hello')



@app.command()
def random_tester(reps, v):
    n = int(reps)
    num_vertices = int(v)
    prob = 0.5
    file_name = 'random.gr'
    for i in range(n):
        Gnx = nx.gnp_random_graph(num_vertices, prob, seed=i, directed=True)
        Gnx = nx.convert_node_labels_to_integers(Gnx, first_label=1)
        nx.write_edgelist(Gnx, file_name, data=False)
        line = f"p tww {len(Gnx.nodes)} {Gnx.number_of_edges()}"
        line_prepender(file_name, line)
        # nx.draw(Gnx)
        print("testing seed", i)
        contraction_tree = get_contraction_tree_from_graph(Gnx)
        g_ver = ver.read_graph(file_name)
        tww = ver.check_sequence(
                g_ver, contraction_tree)
        print(f"twinwidth = {tww}")
    print("All graphs passed test")
    return True




@app.command()
def process_graph_from_file(full_file_name):
    #logging.basicConfig(level=logging.DEBUG)
    instance_path = os.path.dirname(full_file_name)
    output_path = f"output/{os.path.basename(instance_path)}"
    logging.debug(f"output_path : {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.debug(f"Processing file {full_file_name}")
    file_name_without_ext = os.path.splitext(os.path.basename(full_file_name))[0]
    logging.debug(f"file_name_without_ext: {file_name_without_ext}")
    output_file_name = f"{output_path}/{file_name_without_ext}.out"
    g = parser.parse(full_file_name)[0]
    contraction_tree = get_contraction_tree_from_graph(g)
    print_contraction_tree(
         contraction_tree, len(g.nodes), print_to_file=True,
         file_path=output_file_name
    )
    g_ver = ver.read_graph(full_file_name)
    return ver.check_sequence(
                g_ver, contraction_tree)

@app.command()
def process_graphs_from_dir(instance_path: Path, start: int =0, to: int =-1):
    files = from_dir(instance_path)
    tww = {}
    for file_name in files:
        full_file_name = (instance_path / file_name).resolve().as_posix()
        tww[file_name] = process_graph_from_file(full_file_name)
    print(tww)


if __name__ == "__main__":
    app()
