import collections, torch, torchvision, numpy

def load_places_vgg16(weight_file):
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

    state_dict = torch.load(weight_file)
    converted_state_dict = ({
        l: torch.from_numpy(numpy.array(v)).view_as(p)
        for k, v in state_dict.items()
        for l, p in model.named_parameters() if k in l})
    model.load_state_dict(converted_state_dict)

    # TODO: figure out normalizations etc.

    return model
