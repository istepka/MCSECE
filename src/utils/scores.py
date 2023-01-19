import numpy as np
import pandas as pd
import numpy.typing as npt

class ScoreCalculator:

    def __init__(self, data: npt.NDArray | pd.DataFrame, data_predictions: npt.NDArray, cont_ind: npt.NDArray, cat_ind: npt.NDArray, random_seed: int = 2023) -> None:
        '''
        `data`: training data. Cannot containg target columns.
        `data_predictions`: predicted classes for the data.
        `cont_ind`: indices of columns of continous features
        `cat_ind`: indices of columns of categorical features
        '''
        np.random.seed(random_seed)

        self.data = data
        self.data_predictions = data_predictions
        self.cont_ind = cont_ind
        self.cat_ind =  cat_ind

        # Continous data
        self.cont_data = data[:, cont_ind].astype('float64')
        # Categorical data
        self.cat_data = data[:, cat_ind]

        # Set ranges
        self.ranges: npt.NDArray = None
        self.get_ranges()


    def get_ranges(self) -> None:
        '''
        Get ranges for continous variables.
        Return in form of array([min array, max array])
        '''
        mins = self.cont_data.min(axis=0)
        maxes = self.cont_data.max(axis=0)
        self.ranges = maxes - mins


    def heom(self, x: npt.NDArray, y: npt.NDArray) -> float:
        '''
        Calculate HEOM distance between x and y. 
        X and Y should not be normalized. 
        Ranges is max-min on each continous variables (order matters). 
        '''
        distance = 0.0

        # Continous |x-y| / range
        distance += np.sum(np.abs(x[self.cont_ind].astype('float64') - y[self.cont_ind].astype('float64')) / self.ranges)

        # Categorical - overlap
        distance += np.sum(~np.equal(x[self.cat_ind], y[self.cat_ind]))

        return distance


    def implausibility(self, counterfactuals) -> float:
        '''
        Implausibility measures the level of feasibility of the set of counterfactuals C, whether they could be realistic to be realized. 
        The generated counterfactual is realistic in the sense that it will be sufficiently close to the reference (training) data X. 
        It could be defined in many ways. This could be implemented as a distance d between generated cf to their nearest real neighbors from X. 
        The lower average distance, the more preferred counterfactual.
        '''
        pass


    def feasibility(self, cf: npt.NDArray, x: npt.NDArray) -> float:
        '''
        Calculate feasibility as min distance between `cf` any datapoint (different than `x`) from training data.
        Distance metric is HEOM. 

        The lower the better
        '''
        best_d = np.inf
        for y in self.data:
            d = self.heom(cf, y)
            if d < best_d and not np.all(np.equal(x, y)):
                best_d = d
        return best_d
    

    def feasibility_k_neighbors(self, cf: npt.NDArray, x: npt.NDArray, k_neighbors: int = 50) -> float:  
        '''
        Same as feasibility, but averaged over k-nearest-neighbors. So it is sum of distances to k-nearest-neighbors / k.  
        It should aim to measure close the counterfactual is to the training data.

        The lower the better.
        '''
        assert self.data.shape[0] > k_neighbors, "Cannot calculate feasibility_k_neighbors because k_neighbors parameter is greater than the number of datapoints"
        
        distances = list()

        for y in self.data:
            if not np.all(np.equal(x, y)): # Dont add x to the list of distances
                d = self.heom(cf, y)
                distances += [d]
        
        distances.sort()
        top_k = np.array(distances[:k_neighbors], dtype='float64')

        return np.sum(top_k) / k_neighbors


    def features_changed(self, cf: npt.NDArray, x: npt.NDArray, float_precision: float = 1e-5) -> float:
        '''
        Calculate the number of features that changed between counterfactual and original instance.   

        Normalized by number of features -> change_count / count_of_all_features  

        The lower the better
        '''
        fc = 0.0

        # Continous
        fc += np.sum(~np.isclose(cf[self.cont_ind].astype('float64'), x[self.cont_ind].astype('float64'), atol=float_precision))

        # Categorical
        fc += np.sum(~np.equal(cf[self.cat_ind], x[self.cat_ind]))

        return fc / len(cf)


    def proximity(self, cf: npt.NDArray, x: npt.NDArray) -> float:
        '''
        Proxmity is the distance from counterfactual `cf` to its original instance `x`.  

        As a distance function we use HEOM.  

        The lower the better.
        '''
        return self.heom(cf, x)


    def discriminative_power(self, cf: npt.NDArray, cf_predicted_class: npt.NDArray, x: npt.NDArray, x_predicted_class: npt.NDArray, k_neighbors: int = 10) -> float:
        '''
        Reclassification rate of its k nearest neighbors. Neighbors are defined with HEOM distance metric.  

        The higher the better.
        '''
        assert self.data.shape[0] > k_neighbors, "Cannot calculate discriminative power because k_neighbors parameter is greater than the number of datapoints"

        distances = list()

        for y, yclass in zip(self.data, self.data_predictions):
            if not np.all(np.equal(x, y)): # Dont add x to the list of distances
                d = self.heom(cf, y)
                distances += [(d, yclass)]
        
        distances.sort(key=lambda x: x[0])

        # Take predicted classes of top_k distances
        top_k = np.array(distances[:k_neighbors], dtype='float64')[:, 1]
  
        rate = np.sum(top_k == float(cf_predicted_class)) / k_neighbors

        return rate

    def dcg(self, cf: npt.NDArray, x:npt.NDArray, preference_ranking: npt.NDArray) -> float:
        '''
        Calculate the adaptation of dcg metric. Calculate the relevance of feature changes among preferred features.
        Changes calculated as featurewise HEOM.

        `preference_ranking`: array of indices ranked from best to worst. Ranking can be of whatever length.

        The higher the better
        '''

        changes = np.zeros_like(cf, dtype='float64')

        # Continous
        changes[self.cont_ind] = np.abs(cf[self.cont_ind].astype('float64') - x[self.cont_ind].astype('float64')) / self.ranges
        
        # Categorical
        changes[self.cat_ind] = ~np.equal(cf[self.cat_ind], x[self.cat_ind])

        # Calculate DCG score
        dcg_score = 0.0
        for i, index in enumerate(preference_ranking, 1):
            dcg_score += changes[index] / np.log2(i + 1)

        return dcg_score


def get_scores(cfs: npt.NDArray, cf_predicted_classes: npt.NDArray,  
    x: npt.NDArray, x_predicted_class: npt.NDArray,  
    training_data: pd.DataFrame | npt.NDArray, training_data_predicted_classes: npt.NDArray,  
    continous_indices: npt.NDArray, categorical_indices: npt.NDArray,  
    preferences_ranking: npt.NDArray, k_neighbors_feasib: int = 50, 
    k_neighbors_discriminative: int = 20
    ) -> pd.DataFrame:
    '''
    Obtain metrics evaluation for the data.  

    `cfs`: Counterfactuals 
    `cf_predicted_classes`: Counterfactuals predicted classes
    `x`: Original instance corresponding to counterfactuals
    `x_predicted_class`: Original instance predicted classes (corresponding to counterfactuals)
    `trainig_data`: Data to evaluate. Must be in the non-normalized form. Without target columns.  
    `trainig_data_predicted_classes`: 1-D array of predicted classes.  
    `continous_indices`: Column indices of conitnous features.    
    `categorical_indices`: Column indices of categorical features.  
    `preferences_ranking`: Indices of prefered features ranked from best to worst. Length of this array can be whatever. 

    Important: Len of `continous_indices` + `categorical_indices must` be of length `data`
    '''
    assert cfs.shape[1] == training_data.shape[1], 'Counterfactuals and training data have different number of features!'
    assert len(continous_indices) + len(categorical_indices) == cfs.shape[1], 'Designated cat and cont indices should combined have the same length as the counterfactual'
    assert len(cfs) == len(cf_predicted_classes), 'Cfs and cf_predicted classes should have equal lengths'
    assert len(training_data) == len(training_data_predicted_classes), 'trainig_data and trainig_data_predicted_classes  should have equal lengths'

    if isinstance(training_data, pd.DataFrame):
        _training_data = training_data.to_numpy().copy()
    else:
        _training_data = training_data.copy()
    
    # Init score calculator
    calculator = ScoreCalculator(data=_training_data, data_predictions=training_data_predicted_classes, cont_ind=continous_indices, cat_ind=categorical_indices)

    result = list()

    for cf, cf_class in zip(cfs, cf_predicted_classes):

        cf = cf.flatten()

        feasib = calculator.feasibility(cf=cf, x=x)
        print(f'Feasibility: {feasib:.4f}')

        feasib_k = calculator.feasibility_k_neighbors(cf=cf, x=x, k_neighbors=k_neighbors_feasib)
        print(f'Feasibility w.r.t k-neigbors k={k_neighbors_feasib}: {feasib_k:.4f}')

        fc = calculator.features_changed(cf=cf, x=x)
        print(f'Features changed (normalized): {fc:.4f}')

        prox = calculator.proximity(cf=cf, x=x)
        print(f'Proximity: {prox:.4f}')

        disc = calculator.discriminative_power(cf=cf, cf_predicted_class=cf_class, x=x, x_predicted_class=x_predicted_class, k_neighbors=k_neighbors_discriminative)
        print(f'Discriminative power k={k_neighbors_discriminative}: {disc:.4f}')

        dcg = calculator.dcg(cf=cf, x=x, preference_ranking=preferences_ranking)
        print(f'DCG @ {len(preferences_ranking)}: {dcg:.4f}')

        result.append({
            'Proximity': prox,
            'Feasibility': feasib,
            f'Feasibility w.r.t k-neigbors k={k_neighbors_feasib}': feasib_k,
            'Features Changed (normalized)': fc,
            f'Discriminative Power k={k_neighbors_discriminative}': disc,
            f'DCG @{len(preferences_ranking)}': dcg
        })

    scores_df = pd.DataFrame(result)
    return scores_df


if __name__ == '__main__':
    print('-'*30)
    print('Scores calculation exaples: \n')

    c = np.array([5, 2, 'Male', 'Maybe'])
    X = np.array([
        np.array([0, 10, 'Female', 'No']),
        np.array([1, 5, 'Non-specified', 'Yes']),
        np.array([1, 1, 'Male', 'Maybe']),
        np.array([3, 1, 'Female', 'No']),
        np.array([3, 7, 'Female', 'Yes']),
        np.array([2, 1, 'Female', 'No']),
        np.array([3, 8, 'Male', 'No']),
    ])
    classes = np.array([0,0,1,1,1,0,0])
    x = X[2]

    preference = np.array([3,1,2])

    cont_ind = np.array([0, 1])
    cat_ind = np.array([2, 3])

    calculator = ScoreCalculator(data=X, data_predictions=classes, cont_ind=cont_ind, cat_ind=cat_ind)

    feasib = calculator.feasibility(c, x)
    print(f'Feasibility: {feasib:.4f}')

    feasib_k = calculator.feasibility_k_neighbors(cf=c, x=x, k_neighbors=3)
    print(f'Feasibility w.r.t k-neigbors: {feasib_k:.4f}')

    fc = calculator.features_changed(cf=c, x=x)
    print(f'Features changed (normalized): {fc:.4f}')

    prox = calculator.proximity(cf=c, x=x)
    print(f'Proximity: {prox:.4f}')

    disc = calculator.discriminative_power(cf=c, cf_predicted_class=1, x=x, x_predicted_class=0, k_neighbors=3)
    print(f'Discriminative power: {disc:.4f}')

    dcg = calculator.dcg(cf=c, x=x, preference_ranking=preference)
    print(f'DCG@{len(preference)}: {dcg:.4f}')

    cfs = np.array([
        [5,6,'Female', 'Maybe'],
        [1, 1, 'Male', 'Yes']
    ])

    cfs_classes = np.array([1,1])


    scores = get_scores(cfs=cfs, cf_predicted_classes=cfs_classes, 
        x=x, x_predicted_class=0, 
        training_data=X, training_data_predicted_classes=classes,
        continous_indices=cont_ind, categorical_indices=cat_ind, 
        preferences_ranking=preference,
        k_neighbors_discriminative=3, k_neighbors_feasib=3
        )
    
    print(scores.head(10))