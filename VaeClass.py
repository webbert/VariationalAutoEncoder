"""
The following code uses tensorflow and sklearn to create a VAE model class.
VAE model has issues saving and loading which causes issues to reuse the model
to test.

- Issue resides with model.save() function and load() model. 

Created by: Dillon Cheong
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler

LATENT_DIM = 2
HIDDEN_ACTIVATION = 'linear'
OUTPUT_ACTIVATION = 'linear'
LOSS = mse
OPTIMIZER = 'adam'
EPOCHS = 100
BATCH_SIZE = 100
DROPOUT_RATE = 0.2
L2_REGULARIZER = 0.1
VALIDATION_SIZE = 0.2
PREPROCESSING = True
CONTAMINATION = 0.0
GAMMA = 1.0
CAPACITY = 0.0

class VAE():
    def __init__(self, n_features_, neurons):
        """This variational auto encoder has only 2 layers. The number of 
        neurons in the second layer is determined by neurons given divided by 2.

        Args:
            n_features_ (int): No of Features
            neurons (int): No of neurons
        """
        self.n_features = n_features
        self.neurons = neurons
        

    def sample_z(self, args):
        z_mean, z_log = args
        batch = K.shape(z_mean)[0]  # batch size
        dim = K.int_shape(z_mean)[1]  # latent dimension
        epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0
        return z_mean + K.exp(0.5 * z_log) * epsilon

    def vae_loss(self, inputs, outputs, z_mean, z_log, n_features_):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """

        reconstruction_loss = LOSS(inputs, outputs)
        reconstruction_loss *= n_features_
        kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        kl_loss = GAMMA * K.abs(kl_loss - CAPACITY)
        return K.mean(reconstruction_loss + kl_loss)

    def fit(self, x, verbose=0):
        """One class fit which only takes in one data.

        Args:
            x (dataframe/numpy array): takes in usually a numpy array which will
            train the model
            verbose (int, optional): Takes in a number either 0 or 1. 
            Defaults to 0.

        Raises:
            ValueError: If the number of neurons/2 is more than the number of features

        Returns:
            model: Returns a VAE model.
        """
        neurons = self.neurons
        if PREPROCESSING:
            scaler = StandardScaler()
            x_norm = scaler.fit_transform(x)
        anomalies = []

        # BUILDING ENCODER
        if (neurons/2) > self.n_features_:
            raise ValueError("Neurons / 2 cannot be more than n_features.")
        inputs = Input(shape=(self.n_features_,))
        layer = Dense(self.n_features_, activation=HIDDEN_ACTIVATION)(inputs)
        for neurons in [neurons, neurons/2]:
            layer = Dense(neurons, activation=HIDDEN_ACTIVATION,
                            activity_regularizer=l2(L2_REGULARIZER))(layer)
            layer = Dropout(DROPOUT_RATE)(layer)
        z_mean = Dense(LATENT_DIM)(layer)
        z_log = Dense(LATENT_DIM)(layer)
        z = Lambda(sample_z, output_shape=(LATENT_DIM,))([z_mean, z_log])
        encoder = Model(inputs, [z_mean, z_log, z])
        if verbose == 1:
            encoder.summary()

        # BUILDING DECODER
        latent_inputs = Input(shape=(LATENT_DIM,))
        layer = Dense(LATENT_DIM, activation=HIDDEN_ACTIVATION)(latent_inputs)
        for neurons in [neurons/2,neurons]:
            layer = Dense(neurons, activation=HIDDEN_ACTIVATION)(layer)
            layer = Dropout(DROPOUT_RATE)(layer)
        outputs = Dense(self.n_features_, activation=OUTPUT_ACTIVATION)(layer)
        decoder = Model(latent_inputs, outputs)
        if verbose == 1:
            decoder.summary()

        # Compiling VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs)
        vae.add_loss(vae_loss(inputs, outputs, z_mean, z_log, self.n_features_))
        vae.compile(optimizer=OPTIMIZER)
        if verbose == 1:
            vae.summary()

        vae.fit(x_norm, epochs=EPOCHS, batch_size=BATCH_SIZE,
                shuffle=True, validation_split=VALIDATION_SIZE, verbose=False)

        return vae