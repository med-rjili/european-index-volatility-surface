import argparse
from pathlib import Path
import pandas as pd
from volsurface.cleaning import prepare_options

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',default='data/processed/european_options_clean.csv')
    args=p.parse_args()
    df=prepare_options(pd.read_csv(args.input))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print(f'Saved {len(df)} cleaned rows to {out}')
