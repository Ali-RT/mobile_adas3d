"""Restore M63 dataset links/splits, reporting all missing data before mutation."""
import argparse
from pathlib import Path
import shutil

def restore(dataset, splits, candidates):
    dataset, splits = Path(dataset), Path(splits)
    errors, sources, split_sources = [], {}, {}
    for name in ("image_2", "label_2", "calib"):
        aliases = [name] + ([name[:-1] + "02"] if name in ("image_2", "label_2") else [])
        choices = [Path(root)/"training"/alias for root in [dataset, *candidates] for alias in aliases]
        source = next((p for p in choices if p.is_dir()), None)
        if source is None:
            errors.append("Missing " + name + "; checked: " + ", ".join(map(str, choices)))
        else:
            sources[name] = source.resolve()
    for split, count in (("train",3712), ("val",3769)):
        source = splits/(split+".txt")
        if not source.is_file():
            errors.append("Missing split: " + str(source))
            continue
        ids = source.read_text().splitlines()
        if len(ids)!=count or len(set(ids))!=count or any(not i.isdigit() or len(i)!=6 for i in ids):
            errors.append("Invalid split IDs: " + str(source))
            continue
        split_sources[split] = source
        dest = dataset/"ImageSets"/source.name
        if dest.exists() and dest.read_bytes()!=source.read_bytes():
            errors.append("Existing split differs; preserving " + str(dest))
        for name, directory in sources.items():
            suffix = ".png" if name=="image_2" else ".txt"
            missing = [i for i in ids if not (directory/(i+suffix)).is_file()]
            if missing:
                errors.append(f"{split}/{name}: {len(missing)} files missing; first IDs: {missing[:5]}")
    if errors:
        raise RuntimeError("Data preflight failed; no links/splits changed:\n" + "\n".join(errors))
    (dataset/"training").mkdir(parents=True, exist_ok=True)
    (dataset/"ImageSets").mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        dest=dataset/"training"/name
        if dest.is_symlink() and not dest.exists():
            dest.unlink()  # Only replace a dangling local dataset pointer, never its target.
        if not dest.exists():
            dest.symlink_to(source, target_is_directory=True)
    for source in split_sources.values():
        dest=dataset/"ImageSets"/source.name
        if source.resolve()!=dest.resolve():
            shutil.copy2(source,dest)
    print("Data ready: train3712 + val3769 images, labels, calibration and split files.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--splits",type=Path,required=True)
    p.add_argument("--candidate",type=Path,action="append",default=[])
    a=p.parse_args()
    restore(a.dataset,a.splits,a.candidate)

if __name__=="__main__": main()
