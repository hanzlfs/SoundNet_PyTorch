from torch.legacy import nn
from torch.utils.serialization import load_lua
import torch
opt = {
  'dataset' : 'audio',    # indicates what dataset load to use (in data.lua)
  'nThreads' : 40,        # how many threads to pre-fetch data
  'batchSize' : 64,       # self-explanatory
  'loadSize' : 22050*20,  # when loading images, resize first to this size
  'fineSize' : 22050*20,  # crop this size from the loaded image 
  'lr' : 0.001,           # learning rate
  'lambda' : 250,
  'beta1' : 0.9,          # momentum term for adam
  'meanIter' : 0,         # how many iterations to retrieve for mean estimation
  'saveIter' : 5000,      # write check point on this interval
  'niter' : 10000,        # number of iterations through dataset
  'ntrain' : float('inf'),   # how big one epoch should be
  'gpu' : 0,              # which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  'cudnn' : 1,            # whether to use cudnn or not
  'finetune' : '',        # if set, will load this network instead of starting from scratch
  'name' : 'soundnet',    # the name of the experiment
  'randomize' : 1,        # whether to shuffle the data file or not
  'data_root' : '/data/vision/torralba/crossmodal/flickr_videos/soundnet/mp3',
  'label_binary_file' : '/data/vision/torralba/crossmodal/soundnet/features/VGG16_IMNET_TRAIN_B%04d/prob',
  'label2_binary_file' : '/data/vision/torralba/crossmodal/soundnet/features/VGG16_PLACES2_TRAIN_B%04d/prob',
  'label_text_file' : '/data/vision/torralba/crossmodal/soundnet/lmdbs/train_frames4_%04d.txt',
  'label_dim' : 1000,
  'label2_dim' : 401,
  'label_time_steps' : 4,
  'video_frame_time' : 5,  # 5 seconds
  'sample_rate' : 22050,
  'mean' : 0,
}

torch.manual_seed(0)
torch.set_num_threads(1)
torch.set_default_tensor_type('torch.FloatTensor')

global parameters, gradParameters

#### Create data loader

##### create net work

## initialize the model
def weights_init(layer):
    name = torch.typename(layer)
    if name.find('Convolution') > 0 :
        layer.weight.normal_(0.0, 0.01)
        layer.bias.fill_(0)
        #print name, name.find('Convolution')
    elif name.find('BatchNormalization') > 0:
        if layer.weight is not None:
            layer.weight.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.fill_(0)

## create network
def create_network():
    net = nn.Sequential()
    
    net.add(nn.SpatialConvolution(1, 32, 1,64, 1,2, 0,32))
    net.add(nn.SpatialBatchNormalization(32))
    net.add(nn.ReLU(True))
    net.add(nn.SpatialMaxPooling(1,8, 1,8))

    net.add(nn.SpatialConvolution(32, 64, 1,32, 1,2, 0,16))
    net.add(nn.SpatialBatchNormalization(64))
    net.add(nn.ReLU(True))
    net.add(nn.SpatialMaxPooling(1,8, 1,8))

    net.add(nn.SpatialConvolution(64, 128,  1,16, 1,2, 0,8))
    net.add(nn.SpatialBatchNormalization(128))
    net.add(nn.ReLU(True))
    net.add(nn.SpatialMaxPooling(1, 8, 1,8))
    # net.add(nn.SpatialDropout(0.5))

    net.add(nn.SpatialConvolution(128, 256,  1,8, 1,2, 0,4))
    net.add(nn.SpatialBatchNormalization(256))
    net.add(nn.ReLU(True))
    # net.add(nn.SpatialDropout(0.5))

    net.add(nn.ConcatTable().add(nn.SpatialConvolution(256, 1000, 1,16, 1,12, 0,4))
                            .add(nn.SpatialConvolution(256,  401, 1,16, 1,12, 0,4)))

    net.add(nn.ParallelTable().add(nn.SplitTable(3)).add(nn.SplitTable(3)))
    net.add(nn.FlattenTable())
    return net

## -- optimization closure
## the optimizer will call this function to get the gradients
def closure(x):
    gradParameters = gradParameters.zero_()
    ## data_im,data_label,data_label2,data_extra = data:getBatch()
    inputTensor.copy_(data_im.view(opt['batchSize'], 1, opt['fineSize'], 1))
    for i in xrange(opt['label_time_steps']):
        labels[i] = data_label.select(2, i) #labels[i]:copy(data_label:select(3,i)) 

    for i in xrange(opt['label_time_steps']):
        labels[opt['label_time_steps']  - 1 + i] = data_label2.select(2, i)
    
    output = net.forward(inputTensor)
    err = criterion.forward(output, labels) / float(len(labels)) * opt['lambda']
    df_do = criterion.backward(output, labels)
    for i in range(len(labels)):
        df_do.mul(opt['lambda'] / len(labels))
    net.backward(inputTensor, df_do)
    return err, gradParameters # todo : Test this method

def main():
    if opt['gpu'] >= 0 :
        torch.cuda.set_device(opt['gpu'])
    # create or load network
    net = None
    if opt['finetune'] == '' :
        net = create_network()
        output_net = nn.ParallelTable()
        ### output net
        for i in range(8):
            output_net.add(nn.Sequential().add(nn.Contiguous())
                           .add(nn.LogSoftMax()).add(nn.Squeeze()))
        
        net.applyToModules(weights_init)
    else :
        print ('loading :' + opt['finetune'])
        net = load_lua(opt['finetune'])
    #print(net)
    # defind criterion with KL div
    criterion = nn.ParallelCriterion(False)
    for i in range(8):
        criterion.add(nn.DistKLDivCriterion())
    inputs = torch.Tensor(opt['batchSize'], 1, opt['fineSize'], 1).double()
    labels = {}
    for i in range(opt['label_time_steps']):
        labels[i] = torch.Tensor(opt['batchSize'], 1000)
        labels[i + opt['label_time_steps']] = torch.Tensor(opt['batchSize'], 401)
    ## ship everything to GPU if needed
    if opt['gpu'] >= 0 :
        inputs = inputs.cuda()
        for i in range(len(labels)):
            labels[i] = labels[i].cuda()
        net.cuda()
        criterion.cuda()
    ## is not able to conver to cudnn in pytorch so far
    """
    if opt.gpu > 0 and opt.cudnn > 0 then
      require 'cudnn'
      net = cudnn.convert(net, cudnn)
    end

    """
    parameters, gradParameters = net.flattenParameters()
    counter, history = 0, {}
    
