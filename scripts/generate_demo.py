from pathlib import Path
from volsurface.demo import synthetic_chain

if __name__ == '__main__':
    out=Path('data/raw/synthetic_eurostoxx50_options.csv')
    out.parent.mkdir(parents=True,exist_ok=True)
    df=synthetic_chain();df.to_csv(out,index=False)
    print(f'Created {len(df)} explicitly synthetic option quotes: {out}')
