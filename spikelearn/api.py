class Analyzer():

    # PRIVATE METHODS
    def __init__(self,ratNumber, sigma=None, binSize=120, label = None, **kwargs):
        self._ratNumber = ratNumber
        self._sigma = sigma
        self._binSize = binSize
        self._isShuffled = None

        self._trialSpecs = {}
        self._trialRestrictions = {'minDuration':None, 'maxDuration':None,
                                    'trialMin':None,'trialMax':None,
                                    'ntrials':None}


        if label is None:
            self.label = 'Rat '+ str(ratNumber)
        elif 'autoshape' in label:
            self.label = 'rato_%d_autoshape'%ratNumber
        else:
            self.label = label
        self._dset = self._loadFile(**kwargs)
        self.trialsToUse = None

        self.X = None
        self.y = None
        self.trial = None

    def _getData(self,**kwargs):
        pass


    # PUBLIC METHODS

    def selecTrials(self, restrictions, outliers=False,verbose=0,**kwargs):

        '''Select trialsToUse via a restrictions dict of the form {property : value}
        Accepted properties are:
        'minDuration', 'maxDuration' (in milisseconds),
        'trialMin', 'trialMax' (Absolute number of the trial)
        'ntrialsBeg' (turns off last trials to get a total of ntrials)
        '''
        # Reset restrictions
        self._trialRestrictions = {'minDuration':None, 'maxDuration':None,
                                    'trialMin':None,'trialMax':None,
                                    'ntrials':None}

        for propertyi in restrictions:
            self._trialRestrictions[propertyi] = restrictions[propertyi]

        self.trialsToUse = np.ones(self._trialSpecs['Trial number'].shape[0])
        for propertyi in ['minDuration', 'maxDuration','trialMin','trialMax','ntrials']: # enforcing order, leaving ntrials for last
            self.trialsToUse = np.logical_and(self.trialsToUse, self._trialHas(propertyi, self._trialRestrictions[propertyi]) )

        if outliers == 'drop':
            if 'zmax' in kwargs:
                zmax = kwargs['zmax']
            else:
                zmax = 3
            max_zscore_per_trial = np.abs(zscore(np.nanmean(self._dset, axis=1), axis=1)).max(axis=0)
            if verbose >=1: print('There are %d outlier (z>%s) trials being dropped'%(np.sum(max_zscore_per_trial>=zmax),zmax))
            self.trialsToUse[max_zscore_per_trial>=zmax] = 0

        self.X = None
        self.y = None
        self.trial = None

    def manualSelectTrials(self, trialsOfInterest):
        ''' Give an array of booleans of the size of total trials, or an array of integers for the trials of interest'''

        if trialsOfInterest.unique().shape[0] == 2:
            assert trialsOfInterest.shape[0] == self._dset.shape[2]
            self.trialsToUse = trialsOfInterest
        else:
            assert np.sum(np.diff(np.sort(trialsOfInterest))==0) ==0 #there are no repetitions
            self.trialsToUse = np.array([trial in trialsOfInterest for trial in self._trialSpecs['Trial number'] ])


    def selecTimes(self, tmin, tmax=None, z_transform=False,pca=False,**kwargs):
        if self.trialsToUse is None:
            self.trialsToUse = np.ones(self._trialSpecs['Trial number'].shape[0])


        X = np.transpose(self._dset[:,:,self.trialsToUse]).reshape(-1,self._dset.shape[0])
        y = np.arange(X.shape[0])%self._dset.shape[1]
        trial = np.array([self._dset.shape[1]*[t] for t in self._trialSpecs['Trial number'][self.trialsToUse]]).reshape(-1,1)

        if tmax is None:
            tmax = self._trialSpecs['Trial duration'][self.trialsToUse].min()
        if self._binSize == 'norm': # Lets consider the smaller bin, to correctly remove motor activity
            binSize = self._trialSpecs['Trial duration'][self.trialsToUse].min()/10
            leftMostBin = np.ceil(tmin/binSize)
            rightMostBin = np.floor(tmax/binSize)
        else: # in this case, account for the baseline which is 500ms
            leftMostBin = np.ceil((tmin+500)/self._binSize)
            rightMostBin = np.floor((tmax+500)/self._binSize)

        toUseTimes = np.logical_and(y >= leftMostBin, y < rightMostBin)
        self.X = X[toUseTimes,:]

        if self._binSize == 'norm':
            self.y = y[toUseTimes]
        else:
            self.y = y[toUseTimes] - int(500/self._binSize)

        self.trial = trial[toUseTimes].reshape(-1)
        self._isShuffled = False


        if z_transform in ['full','fa']:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

        elif z_transform is 'baseline':
            for trial_i in np.unique(self.trial):
                self.X[self.trial == trial_i,:]

                baseline_mean = self._dset[:,:int(500/self._binSize),trial_i].mean(axis=1).transpose()
                baseline_std = self._dset[:,:int(500/self._binSize),trial_i].std(axis=1).transpose()
                self.X[self.trial == trial_i,:] = (self.X[self.trial == trial_i,:]-baseline_mean)/np.clip(baseline_std,1,None)

        elif z_transform=='robust':
            for neuron in range(self.X.shape[1]):
                meanActivity, stdActivity = huber(self.X[:,neuron],initscale=self.X[:,neuron].std()/3)
                self.X[:,neuron] = (self.X[:,neuron]-meanActivity)/stdActivity

        elif z_transform is False:
            pass
        else:
            raise ValueError('The value {} is not accepted for z_transform'.format(z_transform))



    def shuffleTimes(self):
        for trial_i in np.unique(self.trial):
            self.y[self.trial==trial_i] = np.random.permutation(self.y[self.trial==trial_i])
        self._isShuffled = True

    def cubicNeuronTimeTrial(self, z_transform = False):
        if self._binSize == 'norm':
            useTimes = np.unique(self.y)
        else:
            useTimes = np.unique(self.y) + int(500/self._binSize)

        dsetOfInterest = self._dset[:,:,self.trialsToUse][:,useTimes,:]

        if z_transform:
            for i in range(dsetOfInterest.shape[0]):
                dsetOfInterest[i,:,:] = (dsetOfInterest[i,:,:]- dsetOfInterest[i,:,:].mean())/dsetOfInterest[i,:,:].std()

        return dsetOfInterest

    def describe(self):
        print('Label: %s'%self.label)
        print('Bin size: %s'%self._binSize)
        print('Sigma: %s'%(self._sigma if self._sigma is not None else 'Not smoothed'))

        if self.trialsToUse is not None:
            print('\nUsing %d trials, according to following restrictions:'%self.trialsToUse.sum())
        else:
            print('\nNo selected trials.')
        for restr in [k +': '+ str(self._trialRestrictions[k]) for k in self._trialRestrictions if self._trialRestrictions[k] is not None]:
            print(restr)


        if self.y is not None and self._isShuffled == False:
            bins = np.unique(self.y)
            if self._binSize == 'norm':
                print('Using normalized bins ', bins)
            else:
                print('\nUsing %s time bins:'%len(bins))

            def printBins(bins):
                for b in bins:
                    print('From %s to %dms'%(b*self._binSize, (b+1)*self._binSize ) )
            if len(bins) <= 5:
                printBins(bins)
            else:
                printBins(bins[:3])
                print('.\n.\n.')
                printBins(bins[-2:])
        elif self._isShuffled == True:
            print('The %d bins have been shuffled'%len(np.unique(self.y)))

        else:
            print('\nTime bins not selected.')

    def getTrialsBy(self, ):
        """
        NOT COMPLETED FUNCTION
        DO NOT USE IT
        Options are: first, after_first
        e.g. getTrialsBy({'first':30})
        e.g. getTrialsBy({'after_first':30})
        """
        trialOrder = np.unique(self.trial)

        X = self.X[self.trial< 1]

        return X, y, trial

    def interact(self,realTime=False):
        interactWithActivity(self.cubicNeuronTimeTrial(),realTime=realTime)

    def decode(self, clf, mode, train_size=.5, test_size=.5, init_size=None, n_shuffles=40,
                predict_or_proba='predict', scoring=False,id_kwargs={}, other_rat=None, pca=False):
        """
        Trains and tests the given classifier on data, according to the mode selected.
        Returns a pandas DataFrame with prediction and trial-specific score.
            Parameters of id_kwargs are also saved as identifier variables on the DataFrame.


        Currently supported modes are:
        >>> fixInit_shuffleEnd -> fix an initial subset of trials to test, and shuffle trains on the rest
        >>> fullShuffle -> selects trials at random to fit and predict

        """
        counter = time.time()
        def get_predictions_or_proba(clf, X, mode):
            """
            Local helper function to ease the switching between predict_proba and predict
            """
            if mode == 'predict':
                return clf.predict(X)
            elif mode in ['proba','probability']:
                try:
                    return clf.predict_proba(X)
                except:
                    return clf.decision_function(X)

        if pca:
            assert type(pca) is int
            clf = Pipeline(steps=[('pca', mPCA(n_components=pca) ),('classifier', clf)])
            pca=PCA()
            pca.fit(self.cubicNeuronTimeTrial().mean(axis=2).transpose())

        # Each mode
        if mode in ['fixInit_shuffleEnd','init']:
            results = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true'])
            X_test, y_test, trial_test, X_train, y_train, trial_train =  splitFirstNtrials(self.X, self.y, self.trial, n_before_split = init_size)

            assert not any ([trial in trial_test for trial in trial_train])
            assert init_size is not None

            for i in range(n_shuffles):
                sh_idx = get_n_random_trials_indices(trial_train, train_size)
                clf_local = clone(clf)
                clf_local.fit(X_train[sh_idx,:], y_train[sh_idx])

                true = [y_test[trial_test==ti] for ti in np.unique(trial_test)]
                predictions = [get_predictions_or_proba(clf_local, X_test[trial_test==ti,:], predict_or_proba ) for ti in np.unique(trial_test) ]
                #predictions = [clf_local.predict(X_test[trial_test==ti,:]) for ti in np.unique(trial_test)]
                results = results.append(pd.DataFrame({'shuffle':i, 'predictions': predictions,'true':true, 'trial':np.unique(trial_test)}))

        elif mode is 'fullShuffle':
            X, y, trial = self.X, self.y, self.trial

            results = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true'])
            sh = ShuffleSplit(n_splits=n_shuffles, train_size=train_size,test_size=test_size)

            for i, (train_trials, test_trials) in enumerate(sh.split(np.unique(trial))):
                train_trials = np.unique(trial)[train_trials]
                test_trials = np.unique(trial)[test_trials]

                train_idx = n_to_idx(train_trials, trial)
                clf_local = clone(clf)
                clf_local.fit(X[train_idx,:],y[train_idx])

                assert not any ([tri in test_trials for tri in train_trials])

                predictions = [get_predictions_or_proba(clf_local, X[trial==ti,:], predict_or_proba ) for ti in test_trials ]
                true = [y[trial==ti] for ti in test_trials]
                results = results.append(pd.DataFrame({'shuffle':i, 'predictions': predictions,
                                                        'trial':test_trials, 'true':true}))
        elif mode in ['kfold','kf']:
            X, y, trial = self.X, self.y, self.trial

            results = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true'])
            sh = KFold(n_splits=n_shuffles, shuffle=True)

            for i, (train_trials, test_trials) in enumerate(sh.split(np.unique(trial))):
                train_trials = np.unique(trial)[train_trials]
                test_trials = np.unique(trial)[test_trials]

                train_idx = n_to_idx(train_trials, trial)
                clf_local = clone(clf)
                clf_local.fit(X[train_idx,:],y[train_idx])

                assert not any ([tri in test_trials for tri in train_trials])

                predictions = [get_predictions_or_proba(clf_local, X[trial==ti,:], predict_or_proba ) for ti in test_trials ]
                true = [y[trial==ti] for ti in test_trials]
                results = results.append(pd.DataFrame({'shuffle':i, 'predictions': predictions,
                                                        'trial':test_trials, 'true':true}))


        elif mode in ['cross']:
            assert isinstance(other_rat,Rat)

            # Fit on local, test other
            clf_local = clone(clf)
            clf_local.fit(self.X, self.y)
            predictions = get_predictions_or_proba(clf_local, other_rat.X, predict_or_proba)
            true = [self.y[self.trial==ti] for ti in np.unique(self.trial)]
            results = results.append(pd.DataFrame({'shuffle':i, 'predictions': predictions,
                                                    'trial':test_trials, 'true':true,'trained on':'local','tested on':'other'}))
            # Fit on other, test local

            clf_other = clone(clf)
            clf_other.fit(other_rat.X, other_rat.y)
            get_predictions_or_proba(clf_other, self.X, predict_or_proba)
            true = [other_rat.y[other_rat.trial==ti] for ti in np.unique(other_rat.trial)]
            results = results.append(pd.DataFrame({'shuffle':i, 'predictions': predictions,
                                                    'trial':test_trials, 'true':true,'trained on':'other','tested on':'local'}))


        else:
            raise ValueError('The mode %s is not supported'%mode)

        # Calculate score if requested
        if scoring:
            assert predict_or_proba is 'predict'
            results['kappa'] = results.apply(lambda x: cohen_kappa_score(x['true'],x['predictions'], weights='quadratic'),axis=1)
            results['corr'] = results.apply(lambda x: np.nan_to_num(pearsonr(x['true'],x['predictions'])[0]),axis=1)

        # Add identification labels requested at function call
        for arg in id_kwargs:
            results[arg] = id_kwargs[arg]

        # Add base identification variables
        results['rat_number'] = self._ratNumber
        results['time'] = time.time() - counter
        results['PCA'] = pca

        return results
