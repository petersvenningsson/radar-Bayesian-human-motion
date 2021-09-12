import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class_set = 9


class ObservationsConditionsClassifier():
    """ Container class for several NBGassuian classifiers
    """
    def __init__(self, features, discriminant_model, n_angle_bins):
        self.n_angle_bins = n_angle_bins
        self.features = features

        self.classifiers = [
            ClassifierComposition(self.features, discriminant_model=discriminant_model) for _ in range(self.n_angle_bins)
        ]
        

    def fit(self, df):

        angle_point_estimates = np.vstack(df['angle']).argmax(axis=1)
        df['point_estimate_angle'] = angle_point_estimates
        
        angles = np.stack(df['angle']).argmax(axis=1)
        for i_angle, model in enumerate(self.classifiers):
            samples = df.loc[df['point_estimate_angle'] == i_angle]
            model.fit(samples)


    def predict(self, df):
        
        predictions = self.predict_proba(df)
        return (predictions.argmax(axis=1)+1)


    def predict_proba(self, df):

        angles = np.vstack(df['angle'])
        predictions = np.zeros((len(df), class_set))
        for i_angle, model in enumerate(self.classifiers):
            predictions += (model.predict_proba(df).T * angles[:, i_angle]).T

        return predictions


    def information(self, belief, sensors, n=1000):

        # Sample class membership
        class_samples = np.random.choice(list(range(class_set)), n, p=belief.squeeze())

        sample_probability = np.ones((n, class_set)) # initialization in log scale
        for angle in sensors:
            
            # Sample angle bin
            angle_samples = np.random.choice(list(range(len(angle))), n, p=angle)
            
            # Extract feature distribution for the drawn angle bins
            theta_ = list(map(lambda x: self.classifiers[x].generative_model.model.theta_, angle_samples))
            sigma_ = list(map(lambda x: self.classifiers[x].generative_model.model.sigma_, angle_samples))
            
            # Extract feature distribution for the drawn class
            sample_draw_means = np.array([t[y,:] for (t,y) in zip(theta_, class_samples)])
            sample_draw_std = np.sqrt(np.array([t[y,:] for (t,y) in zip(sigma_, class_samples)]))

            # Draw feature samples
            X = np.random.standard_normal((n, len(self.features)))
            X = X*sample_draw_std + sample_draw_means

            # Build DF to interface with the discriminant classification model
            rows = dict(zip(self.features, X.T))
            rows['lable'] = class_samples

            enc = OneHotEncoder(sparse=False)
            enc.fit(np.array(range(self.n_angle_bins))[:,np.newaxis])
            angle_bins = enc.transform(angle_samples[:, np.newaxis])
            rows['angle'] = [a for a in angle_bins]

            sample_probability += np.log(self.predict_proba(pd.DataFrame(rows)))
        
        enumerator = np.take_along_axis(sample_probability, class_samples[:,None], axis=1).squeeze()
        denomonator = logsumexp((sample_probability + np.repeat(np.log(belief), n, axis=1).T), axis=1)

        information = np.mean(enumerator - denomonator)
        
        return information


class ClassifierComposition:
    """ A composition of a generative model and a discriminant model
    """
    def __init__(self, features, discriminant_model):
        self.priors = [1/class_set for _ in range(class_set)]
        self.discriminant_model = discriminant_model
        self.features = features

        if discriminant_model == 'Gaussian':
            self.model = GaussianNaiveBayes(self.features, self.priors)
            self.generative_model = self.model

        elif discriminant_model == 'calibrated_Gaussian':
            self.model = GaussianNaiveBayes(self.features, self.priors, calibrated=True)
            self.generative_model = GaussianNaiveBayes(self.features, self.priors)

        elif discriminant_model == 'logistic':
            self.model = LogisticReg(self.features)
            self.generative_model = GaussianNaiveBayes(self.features, self.priors)

        else:
            raise ValueError('discriminant model must be in ["Gaussian", "calibrated_Gaussian", "logistic"]')


    def fit(self, df):

        self.model.fit(df)
        self.generative_model.fit(df)


    def predict_proba(self, df):
        return self.model.predict_proba(df)


    def predict(self, df):
        return self.model.predict(df)


    def information(self, belief, sensors, n=1000):
        theta_ = self.generative_model.model.theta_
        sigma_ = self.generative_model.model.sigma_

        class_sample = np.random.choice(list(range(class_set)), n, p=belief.squeeze())

        sample_probability = np.ones((n, class_set))
        for angle in sensors:

            means = np.array(list(map(lambda x: theta_[x,:], class_sample)))
            std = np.sqrt(np.array(list(map(lambda x: sigma_[x,:], class_sample))))

            X = np.random.standard_normal((n, len(self.features)))
            X = X*std + means

            rows = dict(zip(self.features, X.T))
            rows['lable'] = class_sample
            sample_probability += np.log(self.predict_proba(pd.DataFrame(rows)))

        enumerator = np.take_along_axis(sample_probability, class_sample[:,None], axis=1).squeeze()
        denomonator = logsumexp((sample_probability + np.repeat(np.log(belief), n, axis=1).T), axis=1).squeeze()

        information = np.mean(enumerator - denomonator)

        return information


class LogisticReg(LogisticRegression):
    def __init__(self, features):
        super().__init__(class_weight='balanced')
        self.scaler = StandardScaler()
        self.features = features

    def fit(self, df):

        X = df[self.features].to_numpy()
        y = df['lable'].to_numpy()
        X = self.scaler.fit_transform(X)
        return super().fit(X, y)

    def predict(self, df):
        X = df[self.features].to_numpy()
        y = df['lable'].to_numpy()
        X = self.scaler.transform(X)
        return super().predict(X, y)

    def predict_proba(self, df):
        X = df[self.features].to_numpy()
        X = self.scaler.transform(X)
        return super().predict_proba(X)

    def predict_log_proba(self, df):
        X = df[self.features].to_numpy()
        X = self.scaler.transform(X)
        return super().predict_log_proba(X)


class GaussianNaiveBayes():
    def __init__(self, features, priors, calibrated=False):
        self.features = features
        self.priors = priors
        self.calibrated = calibrated

        if calibrated:
            self.model = CalibratedClassifierCV(base_estimator=GaussianNB(priors=self.priors, var_smoothing=1e-6), method='isotonic')
        else:
            self.model = GaussianNB(priors=self.priors, var_smoothing = 1e-6)


    def fit(self, df):

        if self.calibrated:
            # Resample dataset to uniform class distribution
            df = df.groupby('lable', group_keys=False).apply(lambda x: x.sample(len(df), replace=True))
            # GroupShuffleSplit is used to avoid information leak between splits caused by several copies of one sample
            cv = GroupShuffleSplit(n_splits=5).split(df[self.features].to_numpy(), y=df['lable'].to_numpy(), groups=df.index.to_numpy())
            self.model.cv = cv
            
        self.model.fit(df[self.features].to_numpy(), df['lable'].to_numpy())

    def predict(self, df):
        return self.model.predict(df[self.features].to_numpy())

    def predict_proba(self, df):
       return self.model.predict_proba(df[self.features].to_numpy())

    def predict_log_proba(self, df):
        return self.model.predict_log_proba(df[self.features].to_numpy())