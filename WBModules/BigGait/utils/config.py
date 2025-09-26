import yaml


def is_dict(x):
    return isinstance(x, dict)  # or isinstance(x, OrderedDict) or isinstance(x, Odict)


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    # with open("./configs/default.yaml", 'r') as stream:
    #     dst_cfgs = yaml.safe_load(stream)
    # MergeCfgsDict(src_cfgs, dst_cfgs)
    return src_cfgs


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args
