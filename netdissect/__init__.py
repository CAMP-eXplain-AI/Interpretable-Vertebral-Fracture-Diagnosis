'''
Netdissect package.

To run dissection:

1. Load up the convolutional model you wish to dissect, and wrap it
   in an InstrumentedModel.  Call imodel.retain_layers([layernames,..])
   to analyze a specified set of layers.
2. Load the segmentation dataset using the BrodenDataset class;
   use the transform_image argument to normalize images to be
   suitable for the model, or the size argument to truncate the dataset.
3. Write a function to recover the original image (with RGB scaled to
   [0...1]) given a normalized dataset image; ReverseNormalize in this
   package inverts transforms.Normalize for this purpose.
4. Choose a directory in which to write the output, and call
   dissect(outdir, model, dataset).

Example:

    from netdissect import InstrumentedModel, dissect
    from netdissect import BrodenDataset, ReverseNormalize

    model = InstrumentedModel(load_my_model())
    model.eval()
    model.cuda()
    model.retain_layers(['conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
    bds = BrodenDataset('datasets/broden1_227',
            transform_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=1000)
    dissect('result/dissect', model, bds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=10)
'''

__all__ = [
	'actviz',
	'autoeval',
	'bargraph',
	'broden',
	'customnet',
	'easydict',
	'encoder_loss',
	'encoder_net',
	'evalablate',
	'frechet_distance',
	'fsd',
	'fullablate',
	'imgsave',
	'imgviz',
	'invert',
	'LBFGS',
	'make_z_dataset',
	'modelconfig',
	'multilayer_graph',
	'nethook',
	'oldalexnet',
	'oldresnet152',
	'oldvgg16',
	'optimize_residuals',
	'optimize_z_lbfgs',
	'parallelfolder',
	'pbar',
	'pidfile',
	'plotutil',
	'proggan',
	'renormalize',
	'runningstats',
	'samplegan',
	'sampler',
	'segdata',
	'segmenter',
	'segviz',
	'setting',
	'show',
	'statedict',
	'tally',
	'upsample',
	'workerpool',
	'zdataset',
]
