# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger

from pathlib import Path

from smon.main import get_args, main

if __name__ == '__main__':
    ap = get_args()
    ap.add_argument(
        '-p', '--pickle',
        action='store', default=None, type=Path, dest='pkl_fp',
        help='Loads data from a pickled error dump file.'
    )
    ap.add_argument(
        '-l-', '--list-dumps',
        action='store_true', default=False, dest='list_dumps',
        help='List error dump files.'
    )
    args = ap.parse_args().__dict__
    main(**args)
