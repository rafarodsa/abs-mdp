from .printarr import printarr

def parse_oc_args(oc_args):
    assert len(oc_args)%2==0
    oc_args = ['='.join([oc_args[i].split('--')[-1], oc_args[i+1]]) for i in range(len(oc_args)) if i%2==0]
    cli_config = oc.from_cli(oc_args)
    return cli_config