import argparse
from pathlib import Path

parser =  argparse.ArgumentParser()
parser.add_argument('--root-path', type=str, default='/mnt/hdd1/home/joycenerd/AIC21-Track3', help='parent directory path')

args = parser.parse_args()