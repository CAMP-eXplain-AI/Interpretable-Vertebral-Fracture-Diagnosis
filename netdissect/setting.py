import torch, torchvision, os, collections
from . import parallelfolder, zdataset, renormalize, encoder_net, segmenter
from . import bargraph

def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    from . import proggan
    weights_filename = dict(
        bedroom='proggan_bedroom-d8a89ff1.pth',
        church='proggan_churchoutdoor-7e701dd5.pth',
        conferenceroom='proggan_conferenceroom-21e85882.pth',
        diningroom='proggan_diningroom-3aa0ab80.pth',
        kitchen='proggan_kitchen-67f1e16c.pth',
        livingroom='proggan_livingroom-5ef336dd.pth',
        restaurant='proggan_restaurant-b8578299.pth',
        celebhq='proggan_celebhq-620d161c.pth')[domain]
    # Posted here.
    url = 'http://gandissect.csail.mit.edu/models/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    return model

def load_vgg16(domain='places'):
    assert domain == 'places'
    model = torchvision.models.vgg16(num_classes=365)
    model.features = torch.nn.Sequential(collections.OrderedDict(zip([
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'pool1',
        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'pool5'],
        model.features)))
    model.classifier = torch.nn.Sequential(collections.OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))
    baseurl = 'http://gandissect.csail.mit.edu/models/'
    url = baseurl + 'vgg16_places365-6e38b568.pth'
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0

    model.load_state_dict(sd)
    model.eval()
    return model


def load_proggan_ablation(modelname):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)

    from . import proggan_ablation
    model_classname, weights_filename = {
		"equalized-learning-rate": (proggan_ablation.G128_equallr,
            "equalized-learning-rate-88ed833d.pth"),
        "minibatch-discrimination": (proggan_ablation.G128_minibatch_disc,
            "minibatch-discrimination-604c5731.pth"),
        "minibatch-stddev": (proggan_ablation.G128_minibatch_disc,
            "minibatch-stddev-068bc667.pth"),
        "pixelwise-normalization": (proggan_ablation.G128_pixelwisenorm,
            "pixelwise-normalization-4da7e9ce.pth"),
        "progressive-training": (proggan_ablation.G128_simple,
            "progressive-training-70bd90ac.pth"),
        # "revised-training-parameters": (_,
        #     "revised-training-parameters-902f5486.pth")
        "small-minibatch": (proggan_ablation.G128_simple,
            "small-minibatch-04143d18.pth"),
        "wgangp": (proggan_ablation.G128_simple,
            "wgangp-beaa509a.pth")
        }[modelname]
    # Posted here.
    url = 'http://gandissect.csail.mit.edu/models/ablations/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = model_classname()
    model.load_state_dict(sd)
    return model

def load_proggan_inversion(modelname):
    # A couple inversion models pretrained using the code in this repo.

    from . import proggan_ablation
    model_classname, weights_filename = {
		"church": (encoder_net.HybridLayerNormEncoder,
            "church_invert_hybrid_cse-43e52428.pth"),
		"bedroom": (encoder_net.HybridLayerNormEncoder,
            "bedroom_invert_hybrid_cse-b943528e.pth"),
        }[modelname]
    # Posted here.
    url = 'http://gandissect.csail.mit.edu/models/encoders/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('model.', ''): v for k, v in sd.items()}
    model = model_classname()
    model.load_state_dict(sd)
    model.eval()
    return model


g_datasets = {}

def load_dataset(domain, split=None, full=False, download=True):
    if domain in g_datasets:
        return g_datasets[domain]
    if domain == 'places':
        if split is None:
            split = 'val'
        dirname = 'datasets/microimagenet'
        if download and not os.path.exists(dirname):
            os.makedirs('datasets', exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                'http://gandissect.csail.mit.edu/datasets/' +
                'microimagenet.zip',
                'datasets')
        return parallelfolder.ParallelImageFolders([dirname],
                classification=True,
                shuffle=True,
                transform=g_places_transform)
    else:
        # Assume lsun dataset
        if split is None:
            split = 'train'
        dirname = os.path.join(
                'datasets', 'lsun' if full else 'minilsun', domain)
        dirname += '_' + split
        if download and not full and not os.path.exists('datasets/minilsun'):
            os.makedirs('datasets', exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                    'http://gandissect.csail.mit.edu/datasets/minilsun.zip',
                    'datasets',
                    md5='a67a898673a559db95601314b9b51cd5')
        return parallelfolder.ParallelImageFolders([dirname],
                shuffle=True,
                transform=g_transform)

g_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

g_places_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    renormalize.NORMALIZER['imagenet']])

def load_segmenter(segmenter_name='netpqc'):
    '''Loads the segementer.'''
    all_parts = ('p' in segmenter_name)
    quad_seg = ('q' in segmenter_name)
    textures = ('x' in segmenter_name)
    colors = ('c' in segmenter_name)

    segmodels = []
    segmodels.append(segmenter.UnifiedParsingSegmenter(segsizes=[256],
            all_parts=all_parts,
            segdiv=('quad' if quad_seg else None)))
    if textures:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'texture')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="texture", segarch=("resnet18dilated", "ppm_deepsup")))
    if colors:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'color')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="color", segarch=("resnet18dilated", "ppm_deepsup")))
    if len(segmodels) == 1:
        segmodel = segmodels[0]
    else:
        segmodel = segmenter.MergedSegmenter(segmodels)
    seglabels = [l for l, c in segmodel.get_label_and_category_names()[0]]
    segcatlabels = segmodel.get_label_and_category_names()[0]
    return segmodel, seglabels, segcatlabels

def graph_conceptcatlist(conceptcatlist,  cats = None, print_nums = False, **kwargs):
    count = collections.defaultdict(int)
    catcount = collections.defaultdict(int)
    for c in conceptcatlist:
        count[c] += 1
    for c in count.keys():
        catcount[c[1]] += 1
    if cats is None:
        cats = ['object', 'part', 'material', 'texture', 'color']
    catorder = dict((c, i) for i, c in enumerate(cats))
    sorted_labels = sorted(count.keys(),
        key=lambda x: (catorder[x[1]], -count[x]))
    sorted_labels
    tot_num = 0
    if print_nums:
        for k in sorted_labels:
            print(count[k])
            tot_num += count[k]
        print("Total unique concepts: {}".format(tot_num))
    return bargraph.make_svg_bargraph(
        [label for label, cat in sorted_labels],
        [count[k] for k in sorted_labels],
        [(c, catcount[c]) for c in cats], **kwargs)

def save_concept_graph(filename, conceptlist):
    svg = graph_conceptlist(conceptlist, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)

def save_conceptcat_graph(filename, conceptcatlist):
    svg = graph_conceptcatlist(conceptcatlist, barheight=80, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)

def load_test_image(imgnum, split, model, full=False):
    if split == 'gan':
        with torch.no_grad():
            generator = load_proggan(model)
            z = zdataset.z_sample_for_model(generator, size=(imgnum + 1)
                    )[imgnum]
            z = z[None]
            return generator(z), z
    assert split in ['train', 'val']
    ds = load_dataset(model, split, full=full)
    return ds[imgnum][0][None], None

if __name__ == '__main__':
    main()

