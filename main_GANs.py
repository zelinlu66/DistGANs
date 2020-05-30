# -*- coding: utf-8 -*-
"""
@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)

Usage:
  main_GANs.py (-h | --help)
  main_GANs.py --version
  main_GANs.py [--epochs=100] [--optimizer='CGD'] [--learning_rate=0.0005]
  main_GANs.py [-e 100] [-o 'CGD'] [-l 0.0005]
  main_GANs.py [--model='MLP']
  main_GANs.py [-m 'CNN'] [--display] [--save]

  
Options:
  -h, --help                  Show this screen.
  --version                   Show version.
  --display                   Use matplotlib to plot results.
  --save                      Save the model.
  -e, --epochs=<n>            Number of epochs [default: 100].
  -o, --optimizer=<str>       Optimizer name [default: Jacobi].
  -l, --learning_rate=<f>     Learning rate [default: 0.01].
  -m, --model=<str>           Implementation of GANs model. Multi-layer 
                              perceptrons NN (MLP), convolutional NN (CNN) 
                              [default: MLP].
"""
import sys
import matplotlib.pyplot as plt

from docopt import docopt
from MLP_GANs_object import *
from DCGANs_object import *

"""
! READ ME !
Multi-layer perceptrons neural networks (MLP), convolutional neural networks (CNN) 
X = Generator
Y = Discriminaror

BCEWithLogitsLoss is the default cost function, to change it to standard BCE the code needs to be changed
in line 68 of GANs_object or line 67 of DCGANs_object

Different learning rates for X and Y can only be used with 'Jacobi' and 'JacobiMultiCost'
for the other optimizers the learning rate will be set to the value of lr_x

Label smoothing variation is implemented only for optimizer 'Jacobi' and only for GANs_object

To swich from MLP_GANs (DCGANs) to DCGANs (MLP_GANs) comment lines 38,39,40 (43,44,45) and uncomment 43,44,45 (38,39,40)

Attribute save_models of both training object saves the state dicts of the networks into 2 different folders
inside your current directory

TO TRY: Setting much lower learning rates to see if model collapse is avoided (ex. lr_x = 0.0001 , lr_y = 0.0004)

"""

if __name__ == "__main__":
    arguments = docopt(__doc__, version="Competitive Gradient Descent 0.0")
    print("Input parameters:")
    for arg in arguments:
        if arguments[arg]:
            print("  {}  {}".format(arg, arguments[arg]))

    epochs = int(arguments["--epochs"])
    optimizer_name = arguments["--optimizer"]
    learning_rate = float(arguments["--learning_rate"])

    model_switch = arguments["--model"]

    if model_switch == "MLP":
        print("Using MLP implementation of GANs: MLP_GANs_model")
        model = MLP_GANs_model(mnist_data(rand_rotation=False, max_degree=90))
    elif model_switch == "CNN":
        print("Using CNN implementation of GANs: DCGANs_model")
        model = DCGANs_model(mnist_data_dcgans(rand_rotation=False, max_degree=90))
    else:
        sys.exit(
            "\n   *** Error. Specified model name: {} is not valid. Please choose MLP or CNN".format(
                model_switch
            )
        )

    model.train(
        num_epochs=epochs,
        lr_x=torch.tensor([learning_rate]),
        lr_y=torch.tensor([learning_rate]),
        optimizer_name=optimizer_name,
        verbose=True,
        label_smoothing=False,
        single_number=4,
    )  # save_path = ''

    if arguments["--save"]:
        print("Saving the model...")
        model.save_models()

    if arguments["--display"]:
        plt.figure()
        plt.plot(
            [x for x in range(0, len(model.D_error_real_history))],
            model.D_error_real_history,
        )
        plt.plot(
            [x for x in range(0, len(model.D_error_fake_history))],
            model.D_error_fake_history,
        )
        plt.plot(
            [x for x in range(0, len(model.G_error_history))], model.G_error_history,
        )
        plt.xlabel("Iterations")
        plt.ylabel("Loss function value")
        plt.legend(
            [
                "Discriminator: Loss on Real Data",
                "Discriminator: Loss on Fake Data",
                "Generator: Loss",
            ]
        )
        plt.savefig("cost_report.png")
