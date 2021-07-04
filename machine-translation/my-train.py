import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
import numpy as np
import mxnet as mx
import horovod.mxnet as hvd
import gluonnlp as nlp
import nmt
import dataprocessor
import logging
import argparse
import time
import utils
import random
import hyperparameters as hparams                                                                                                                                                                                                                                                                                                                                                           

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
#ctx = mx.cpu(0)

# Training settings
parser = argparse.ArgumentParser(description='Tranformer')

parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='disable training on GPU (default: False)')
#args = parser.parse_args()
args, unknown = parser.parse_known_args()

#args, _  = parser.parse_known_args()
#hparams={
#        'src_lang' : 'en',
#        'tgt_lang' : 'de'
#        }

#args.no_cuda = False

if not args.no_cuda:
    # Disable CUDA if there are no GPUs.
    if mx.context.num_gpus() == 0:
        args.no_cuda = True

logging.basicConfig(level=logging.INFO)
logging.info(args)

# Initialize Horovod
hvd.init()

# Set context to current process 
ctx = mx.cpu(hvd.rank()) if args.no_cuda else mx.gpu(hvd.local_rank()) 
 

num_workers = hvd.size()

'''wmt_model_name = 'transformer_en_de_512'
wmt_transformer_model, wmt_src_vocab, wmt_tgt_vocab = \
    nlp.model.get_model(wmt_model_name,
                              dataset_name='WMT2014',
                              pretrained=True,
                              ctx=ctx)
print(wmt_src_vocab)
print(wmt_tgt_vocab)'''

# Build model
'''encoder, decoder = nmt.transformer.get_transformer_encoder_decoder(units=hparams.num_units,
                                                   hidden_size=hparams.hidden_size,
                                                   dropout=hparams.dropout,
                                                   num_layers=hparams.num_layers,
                                                   num_heads=hparams.num_heads,
                                                   max_src_length=530,
                                                   max_tgt_length=549,
                                                   scaled=hparams.scaled)
model = nmt.translation.NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 share_embed=True, embed_size=hparams.num_units, tie_weights=True,
                 embed_initializer=None, prefix='transformer_')
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
model.hybridize()'''

'''wmt_data_test = nlp.data.WMT2014BPE('newstest2014',
                                    src_lang=hparams.src_lang,
                                    tgt_lang=hparams.tgt_lang,
                                    full=False)
print('Source language %s, Target language %s' % (hparams.src_lang, hparams.tgt_lang))
wmt_data_test[0]

wmt_test_text = nlp.data.WMT2014('newstest2014',
                                 src_lang=hparams.src_lang,
                                 tgt_lang=hparams.tgt_lang,
                                 full=False)
wmt_test_text[0]'''

demo = True
if demo:
    dataset = 'TOY'
else:
    dataset = 'WMT2014BPE'

data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab = \
    dataprocessor.load_translation_data(
        dataset=dataset,
        src_lang=hparams.src_lang,
        tgt_lang=hparams.tgt_lang)

data_train_lengths = dataprocessor.get_data_lengths(data_train)
data_val_lengths = dataprocessor.get_data_lengths(data_val)
data_test_lengths = dataprocessor.get_data_lengths(data_test)

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_test)])

# Create optimizer
optimizer_params = {'momentum': args.momentum,
                    'learning_rate': args.lr * hvd.size()}
opt = mx.optimizer.create('sgd', **optimizer_params)

train_batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Stack(dtype='float32'),
    nlp.data.batchify.Stack(dtype='float32'))
test_batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Stack(dtype='float32'),
    nlp.data.batchify.Stack(dtype='float32'),
    nlp.data.batchify.Stack())

target_val_lengths = list(map(lambda x: x[-1], data_val_lengths))
target_test_lengths = list(map(lambda x: x[-1], data_test_lengths))

bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                             batch_size=hparams.batch_size,
                                             num_buckets=hparams.num_buckets,
                                             ratio=0.0,
                                             shuffle=True,
                                             use_average_length=True,
                                             num_shards=1,
                                             bucket_scheme=bucket_scheme)
#print('Train Batch Sampler:')
#print(train_batch_sampler.stats())

val_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_val_lengths,
                                       batch_size=hparams.test_batch_size,
                                       num_buckets=hparams.num_buckets,
                                       ratio=0.0,
                                       shuffle=False,
                                       use_average_length=True,
                                       bucket_scheme=bucket_scheme)
#print('Validation Batch Sampler:')
#print(val_batch_sampler.stats())

test_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_test_lengths,
                                        batch_size=hparams.test_batch_size,
                                        num_buckets=hparams.num_buckets,
                                        ratio=0.0,
                                        shuffle=False,
                                        use_average_length=True,
                                        bucket_scheme=bucket_scheme)
#print('Test Batch Sampler:')
#print(test_batch_sampler.stats())

train_data_loader = nlp.data.ShardedDataLoader(data_train,
                                      batch_sampler=train_batch_sampler,
                                      batchify_fn=train_batchify_fn,
                                      num_workers=8)
#print('Length of train_data_loader: %d' % len(train_data_loader))
val_data_loader = gluon.data.DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=test_batchify_fn,
                             num_workers=8)
#print('Length of val_data_loader: %d' % len(val_data_loader))
test_data_loader = gluon.data.DataLoader(data_test,
                              batch_sampler=test_batch_sampler,
                              batchify_fn=test_batchify_fn,
                              num_workers=8)
#print('Length of test_data_loader: %d' % len(test_data_loader))

# Build model
encoder, decoder, one_step_ahead_decoder = nlp.model.transformer.get_transformer_encoder_decoder(units=hparams.num_units,
                                                   hidden_size=hparams.hidden_size,
                                                   dropout=hparams.dropout,
                                                   num_layers=hparams.num_layers,
                                                   num_heads=hparams.num_heads,
                                                   max_src_length=530,
                                                   max_tgt_length=549,
                                                   scaled=hparams.scaled)
model = nlp.model.translation.NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 one_step_ahead_decoder=one_step_ahead_decoder, share_embed=False, embed_size=hparams.num_units, 
                                       tie_weights=False, embed_initializer=None, prefix='transformer_')
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
model.hybridize()

#print(model)

# Fetch and broadcast parameters
params = model.collect_params()
if params is not None:
    hvd.broadcast_parameters(params, root_rank=0)

# Create DistributedTrainer, a subclass of gluon.Trainer
trainer = hvd.DistributedTrainer(params,opt)

label_smoothing = nlp.loss.LabelSmoothing(epsilon=hparams.epsilon, units=len(tgt_vocab))
label_smoothing.hybridize()

loss_function = nlp.loss.MaskedSoftmaxCrossEntropyLoss(sparse_label=False)
loss_function.hybridize()

#test_loss_function = nlp.loss.MaskedSoftmaxCrossEntropyLoss()
test_loss_function =  nlp.loss.MaskedSoftmaxCELoss()
test_loss_function.hybridize()

detokenizer = nlp.data.SacreMosesDetokenizer()

translator = nmt.translation.BeamSearchTranslator(model=model,
                                                  beam_size=hparams.beam_size,
                                                  scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha,
                                                                                    K=hparams.lp_k),
                                                  max_length=200)
#print('Use beam_size=%d, alpha=%.2f, K=%d' % (hparams.beam_size, hparams.lp_alpha, hparams.lp_k))

trainer = gluon.Trainer(model.collect_params(), hparams.optimizer,
                        {'learning_rate': hparams.lr, 'beta2': 0.98, 'epsilon': 1e-9})
#print('Use learning_rate=%.2f' % (trainer.learning_rate))

# Train model
best_valid_loss = float('Inf')
step_num = 0
#We use warmup steps as introduced in [1].
warmup_steps = hparams.warmup_steps
grad_interval = hparams.num_accumulated
model.collect_params().setattr('grad_req', 'add')
#We use Averaging SGD [2] to update the parameters.
average_start = (len(train_data_loader) // grad_interval) * \
    (hparams.epochs - hparams.average_start)
average_param_dict = {k: mx.nd.array([0]) for k, v in
                                      model.collect_params().items()}
update_average_param_dict = True
model.collect_params().zero_grad()
print('Numero de Epochs = %d' % (hparams.epochs))
init_time = time.time()
for epoch_id in range(hparams.epochs):
    print('Epoch %d - train_one_epoch started' % (epoch_id))
    utils.train_one_epoch(epoch_id, model, train_data_loader, trainer,
                          label_smoothing, loss_function, grad_interval,
                          average_param_dict, update_average_param_dict,
                          step_num, ctx, hvd.rank(), init_time)
    print('Epoch %d - train_one_epoch finished' % (epoch_id))
    mx.nd.waitall()
    print('Epoch %d - after waitall' % (epoch_id))
    # We define evaluation function as follows. The `evaluate` function use beam search translator
    # to generate outputs for the validation and testing datasets.
    '''valid_loss, _ = utils.evaluate(model, val_data_loader,
                                   test_loss_function, translator,
                                   tgt_vocab, detokenizer, ctx)
    print('Epoch %d, valid Loss=%.4f, valid ppl=%.4f'
          % (epoch_id, valid_loss, np.exp(valid_loss)))
    test_loss, _ = utils.evaluate(model, test_data_loader,
                                  test_loss_function, translator,
                                  tgt_vocab, detokenizer, ctx)
    print('Epoch %d, test Loss=%.4f, test ppl=%.4f'
          % (epoch_id, test_loss, np.exp(test_loss)))
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model.save_parameters('{}.{}'.format(hparams.save_dir, 'valid_best.params'))'''
    model.save_parameters('{}.epoch{:d}.params'.format(hparams.save_dir, epoch_id))
mx.nd.save('{}.{}'.format(hparams.save_dir, 'average.params'), average_param_dict)

'''if hparams.average_start > 0:
    for k, v in model.collect_params().items():
        v.set_data(average_param_dict[k])
else:
    model.load_parameters('{}.{}'.format(hparams.save_dir, 'valid_best.params'), ctx)
valid_loss, _ = utils.evaluate(model, val_data_loader,
                               test_loss_function, translator,
                               tgt_vocab, detokenizer, ctx)
print('Best model valid Loss=%.4f, valid ppl=%.4f'
      % (valid_loss, np.exp(valid_loss)))
test_loss, _ = utils.evaluate(model, test_data_loader,
                              test_loss_function, translator,
                              tgt_vocab, detokenizer, ctx)
print('Best model test Loss=%.4f, test ppl=%.4f'
      % (test_loss, np.exp(test_loss)))'''