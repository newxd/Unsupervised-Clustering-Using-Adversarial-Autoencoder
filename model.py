import cPickle as pickle
from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DenseLayer
from lasagne.layers import get_all_layers
from lasagne.layers import get_all_params
from lasagne.nonlinearities import linear, rectify, sigmoid,softmax


def save_weights(weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_weights(layer, filename):
    with open(filename, 'rb') as f:
        src_params_list = pickle.load(f)

    dst_params_list = get_all_params(layer)
    # assign the parameter values stored on disk to the model
    for src_params, dst_params in zip(src_params_list, dst_params_list):
        dst_params.set_value(src_params)


def build_model():
    num_input = 3
    # should really use more dimensions, but this is nice for visualization
    num_code = 3
    num_hidden = 10

    l_encoder_in = InputLayer((None, num_input), name='l_encoder_in')

    # first layer of the encoder/generator
    l_dense1 = DenseLayer(
        l_encoder_in, num_units=num_hidden, nonlinearity=rectify,
        name='l_encoder_dense1',
    )
    l_dense1.params[l_dense1.W].add('generator1')
    l_dense1.params[l_dense1.b].add('generator1')
    l_dense1.params[l_dense1.W].add('generator2')
    l_dense1.params[l_dense1.b].add('generator2')

    # second layer of the encoder/generator
    l_dense2 = DenseLayer(
        l_dense1, num_units=num_hidden, nonlinearity=rectify,
        name='l_encoder_dense2',
    )
    l_dense2.params[l_dense2.W].add('generator1')
    l_dense2.params[l_dense2.b].add('generator1')
    l_dense2.params[l_dense2.W].add('generator2')
    l_dense2.params[l_dense2.b].add('generator2')

    # output of the encoder/generator: q(z|x)
    l_encoder_out1 = DenseLayer(
        l_dense2, num_units=2, nonlinearity=softmax,
        name='l_encoder_out1',
    )
    l_encoder_out1.params[l_encoder_out1.W].add('generator1')
    l_encoder_out1.params[l_encoder_out1.b].add('generator1')


    l_encoder_out2 = DenseLayer(
        l_dense2, num_units=num_code, nonlinearity=linear,
        name='l_encoder_out2',
    )
    l_encoder_out2.params[l_encoder_out2.W].add('generator2')
    l_encoder_out2.params[l_encoder_out2.b].add('generator2')
    l_encoder_out=ConcatLayer([l_encoder_out1,l_encoder_out2],name='l_encoder_out')
    #l_encoder_out.params[l_encoder_out.W].add('generator')
    #l_encoder_out.params[l_encoder_out.b].add('generator')

    # first layer of the decoder
    l_decoder_in = DenseLayer(
        l_encoder_out, num_units=num_hidden, nonlinearity=rectify,
        name='l_decoder_dense1',
    )
    # second layer of the decoder
    l_dense5 = DenseLayer(
        l_decoder_in, num_units=num_hidden, nonlinearity=rectify,
        name='l_decoder_dense2',
    )

    # output of the decoder: p(x|z)
    l_decoder_out = DenseLayer(
        l_dense5, num_units=num_input, nonlinearity=sigmoid,
        name='l_decoder_out',
    )

    # input layer providing samples from p(z)
    l_prior1 = InputLayer((None, 2), name='l_prior_in_1')

    # concatenate samples from q(z|x) to samples from p(z)
    l_concat1 = ConcatLayer(
        [l_encoder_out1, l_prior1], axis=0, name='l_prior_encoder_concat1',
    )

    # first layer of the discriminator
    l_dense6_softmax= DenseLayer(
        l_concat1, num_units=num_hidden, nonlinearity=rectify,
        name='l_discriminator_dense1_softmax',
    )
    l_dense6_softmax.params[l_dense6_softmax.W].add('discriminator1')
    l_dense6_softmax.params[l_dense6_softmax.b].add('discriminator1')

    # second layer of the discriminator
    l_dense7_softmax = DenseLayer(
        l_dense6_softmax, num_units=num_hidden, nonlinearity=rectify,
        name='l_discriminator_dense2_softmax',
    )
    l_dense7_softmax.params[l_dense7_softmax.W].add('discriminator1')
    l_dense7_softmax.params[l_dense7_softmax.b].add('discriminator1')

    # output layer of the discriminator
    l_discriminator_out1 = DenseLayer(
        l_dense7_softmax, num_units=1, nonlinearity=sigmoid,
        name='l_discriminator_out1',
    )
    l_discriminator_out1.params[l_discriminator_out1.W].add('discriminator1')
    l_discriminator_out1.params[l_discriminator_out1.b].add('discriminator1')





    # input layer providing samples from p(z)
    l_prior2 = InputLayer((None, num_code), name='l_prior_in_2')

    # concatenate samples from q(z|x) to samples from p(z)
    l_concat2 = ConcatLayer(
        [l_encoder_out2, l_prior2], axis=0, name='l_prior_encoder_concat2',
    )

    # first layer of the discriminator
    l_dense6 = DenseLayer(
        l_concat2, num_units=num_hidden, nonlinearity=rectify,
        name='l_discriminator_dense1',
    )
    l_dense6.params[l_dense6.W].add('discriminator2')
    l_dense6.params[l_dense6.b].add('discriminator2')

    # second layer of the discriminator
    l_dense7 = DenseLayer(
        l_dense6, num_units=num_hidden, nonlinearity=rectify,
        name='l_discriminator_dense2',
    )
    l_dense7.params[l_dense7.W].add('discriminator2')
    l_dense7.params[l_dense7.b].add('discriminator2')

    # output layer of the discriminator
    l_discriminator_out2 = DenseLayer(
        l_dense7, num_units=1, nonlinearity=sigmoid,
        name='l_discriminator_out2',
    )
    l_discriminator_out2.params[l_discriminator_out2.W].add('discriminator2')
    l_discriminator_out2.params[l_discriminator_out2.b].add('discriminator2')


    model_layers = get_all_layers([l_decoder_out, l_discriminator_out1,l_discriminator_out2])

    # put all layers in a dictionary for convenience
    return {layer.name: layer for layer in model_layers}


if __name__ == '__main__':
    layer_dict = build_model()
    print('collected %d layers' % (len(layer_dict.keys())))
    for name in layer_dict:
        print('%s: %r' % (name, layer_dict[name]))
