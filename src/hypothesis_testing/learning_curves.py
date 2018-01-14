from sklearn.decomposition import PCA



# TODO Neuron number curve
    # TODO both autoshape and after


for rat_number in [7,8,9,10]:
    auto = Rat(rat_number, sigma=None, binSize=120, label='autoshape', method = 32)
    true = Rat(rat_number, sigma=None, binSize=120, method = 32)
    boot = Rat(rat_number, sigma=None, binSize=120, method = 32)

    for rat in [auto, true, boot]:
        rat.selecTrials({'trialMax':'best', 'minDuration':1500})

    for n_neurons in range(1,33):
        for rat in [auto, true, boot]:
            rat._dset = rat._dset[:n_neurons,:,:]
            rat.selecTimes(200,1200)

            predictions = rat.decode('kfold', 10, )


# TODO SCA learning curve
# TODO PCA learning curve
