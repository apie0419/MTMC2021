import argparse
from pathlib import Path

parser =  argparse.ArgumentParser()
parser.add_argument('--root-path', type=str, default='/home/apie/projects/MTMC2021_ver2/dataset', help='parent directory path')

args = parser.parse_args()