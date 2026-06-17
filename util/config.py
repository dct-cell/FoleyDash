from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree

from omegaconf import DictConfig, OmegaConf

parser = ArgumentParser()
parser.add_argument("--rundir", type=str, default=None)
args = parser.parse_args()


def load_config(*files: str, load_root=False) -> DictConfig:
    """
    Load YAML files[0], files[1], ... and merge them.
    load_root:  If False, load files from get_rundir()
                If True, load files from "./"
    """
    confs = []
    rundir = Path("./") if load_root else get_rundir()
    for file in files:
        # Loading {rundir}/{file}.yaml
        path = (rundir / "cfg" / file).with_suffix(".yaml")
        try:
            confs.append(OmegaConf.load(path))
        except FileNotFoundError:
            print(f"Config YAML file {path} not found. Skipped.")
        except Exception as e:
            print(f"Error loading config YAML file {path}: {e}")
    return OmegaConf.merge(*confs)


def get_rundir():
    """
    Returns --rundir (if exists) or rundir in the YAML file in "./".
    If in ./base.yaml rundir is set with ${foo}, then it'll be parsed by OmegaConf
    as long as some key in ./base.yaml or ./train.yaml is set with a value
    """
    if args.rundir is None:
        cfg = load_config("base", "train", load_root=True)
        rundir = Path(cfg.rundir)
    else:
        rundir = Path(args.rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    return rundir


def backup_cfg():
    """
    Copy config files from ./cfg to ${rundir}/cfg
    """
    if args.rundir is not None:
        # starting an existing specific experiment
        return
    dest_cfg_dir = get_rundir() / "cfg"
    if dest_cfg_dir.exists():
        # restarting an experiment specified in cfg
        return
    try:
        copytree("./cfg", dest_cfg_dir)
        print(f"Config backup success. You can modify config files now.")
    except Exception as e:
        print(f"Error occurred during config backup: {e}")
