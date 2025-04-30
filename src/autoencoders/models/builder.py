from . import vgg, resnet

def BuildAutoEncoder(args):
    img_size = 224

    if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.arch)
        model = vgg.VGGAutoEncoder(configs, img_size=img_size)

    elif args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck, img_size=img_size)
    
    else:
        return None

    return model.cuda()