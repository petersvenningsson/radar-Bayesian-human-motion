import argparse

import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import dataloader
import track
from classifiers import ObservationsConditionsClassifier
from classifiers import ClassifierComposition

np.seterr(all='ignore')

class_set = 9
n_pca_components = 20

def train():
    global parsed_args
    test_sequence = 'Mix'
    measurement_costs = [0.1*i for i in range(0,15)]
    measurement_costs.extend([0.01*i for i in range(1, 15)])

    loader = dataloader.DataLoaderSpectrogram()
    features = [f'PC_{i}' for i in range(n_pca_components)]

    classifiers = [
        (ObservationsConditionsClassifier(features, discriminant_model='calibrated_Gaussian', n_angle_bins=8), 'Conditioned on $\phi$', 'Isotonic calibration'),
        (ObservationsConditionsClassifier(features, discriminant_model='Gaussian', n_angle_bins=8), 'Conditioned on $\phi$','Uncalibrated'),
        (ClassifierComposition(features, discriminant_model='Gaussian'), 'Not conditioned on $\phi$', 'Uncalibrated'),
        (ClassifierComposition(features, discriminant_model='calibrated_Gaussian'), 'Not conditioned on $\phi$', 'Isotonic calibration'),
    ]

    rows = []
    for cost in measurement_costs:
        for i_model, (classifier, observation_condition, discriminant_model) in enumerate(classifiers):

            if parsed_args.rebuild:
                track.state_estimation(load_directory = './data/dataset/RD')
                dataset_path = r'C:\Users\peter\Documents\pulseON'
                loader = dataloader.DataLoaderSpectrogram()
                loader.build(dataset_path,'PCA')
            else:
                loader.load('./data/dataset_df')

            result_df = evaluate_classifier(classifier, loader.df, test_persons = loader.df.person.unique(), test_sequence = test_sequence, measurement_cost = cost)

            predictions = result_df.loc[result_df['sequence_type'] == test_sequence]['prediction'].to_numpy()
            lables = result_df.loc[result_df['sequence_type'] == test_sequence]['lable'].to_numpy()
            accuracy = accuracy_score(lables, predictions)

            rows.append({
                'Accuracy': accuracy, 'Balanced accuracy': balanced_accuracy_score(lables, predictions), 'Macro-averaged Jaccard index': jaccard_score(lables, predictions, average='macro'),
                'Observation conditions': observation_condition, 'Calibration': discriminant_model, 'Cost': cost,
                'result_df': result_df, 'model_index': i_model,
            })
        
    sns.lineplot(data = pd.DataFrame(rows), x = 'Cost', y = 'Accuracy', style = 'Observation conditions', hue = 'Calibration')
    plt.tight_layout()
    plt.show()


def evaluate_classifier(model, df, test_persons, measurement_cost, test_sequence = 'Mix', prior = [1/class_set for i in range(class_set)], render_seq=False):

    df['prediction'] = -6666
    for test_person in test_persons:

        training_df = df.loc[df['person'] != test_person]
        test_df = df.loc[(df['person'] == test_person) & (df['sequence_type'] == test_sequence)].copy()

        transition_matrix = estimate_transition_matrix(
            training_df.loc[training_df['sequence_type'] == 'Mix']
        )

        model.fit(training_df)
        for j, file in enumerate(test_df.file_index.unique()):
            print(f'File {j}/{len(test_df.file_index.unique())}')

            seq_df = test_df.loc[test_df['file_index'] == file].copy()
            seq_df = predict_sequence(model, seq_df, transition_matrix, measurement_cost)
            
            if render_seq:
                render.render_classification_sequence(seq_df)

            df.loc[seq_df.index, 'belief'] = seq_df['belief']
            df.loc[seq_df.index, 'prediction'] = seq_df['prediction']
            df.loc[seq_df.index, 'Selected'] = seq_df['Selected']

    return df


def predict_sequence(model, df, transition_matrix, measurement_cost, prior=[1/class_set for _ in range(class_set)]):
    belief = np.reshape(prior, (class_set, 1))

    for time in np.sort(df.time.unique()):
        df_step = df[df['time'] == time].copy()

        if measurement_cost:
            selected_sensors = information_selection(df_step, model, belief, measurement_cost)
        else:
            selected_sensors = df_step.index
        df.loc[selected_sensors, 'Selected'] = True

        for i, row in df_step.loc[selected_sensors].iterrows():
            row = row.to_frame().transpose()
            prop_likelihood = model.predict_proba(row)
            posterior = prop_likelihood[0, :, np.newaxis] * belief
            posterior = posterior/(posterior.sum())
            belief = posterior
        
        # save prediction
        df['belief'] = np.nan
        df['belief'] = df['belief'].astype(object)
        for index in df_step.index:
            df.loc[index, 'belief'] = [belief]
            df.loc[index ,'prediction'] = belief.argmax() + 1

        # Transition step
        belief = transition_matrix @ np.reshape(belief, (class_set,1))

    return df


def information_selection(df, model, belief, measurement_cost):

    # Calculate information and sort indices by information
    df['information'] = df.apply(lambda row: model.information(belief, [row['predicted_angle']]), axis=1)
    potential_sensors = df.sort_values('information').index.to_list()

    selected_sensors = []
    sensor_utility = {0:[]}
    while potential_sensors:
        selected_sensors.append(potential_sensors.pop())

        information = model.information(belief, sensors=df.loc[selected_sensors]['predicted_angle'].to_list())
        utility = information - measurement_cost*len(selected_sensors)

        sensor_utility[utility] = selected_sensors[:]

    return sensor_utility[np.max(list(sensor_utility.keys()))]


def estimate_transition_matrix(df):
    transition_count = np.zeros((class_set,class_set))
    df = df.loc[df['radar'] == df.radar.unique()[0]]
    sequences = df['file_index'].unique()
    for sequence_index in sequences:
        df_seq = df.loc[df['file_index'] == sequence_index].sort_values('time').reset_index(drop=True)

        previous_state = None
        for i, row in df_seq.iterrows():
            state = row['lable']
            if not previous_state:
                previous_state = state
                continue
            transition_count[state - 1, previous_state - 1] += 1 
            previous_state = state
    transition_matrix = transition_count/transition_count.sum(axis=0,keepdims=1)
    transition_matrix = transition_matrix/transition_matrix.sum(axis=0,keepdims=1)

    return transition_matrix


def load_options():
    global parsed_args
    parser = argparse.ArgumentParser(description='Entry point to fit and evaluate\
                                    a Bayesian model of human motion',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rebuild', dest='rebuild', action='store_true')
    parser.add_argument('--no-rebuild', dest='rebuild', action='store_false')
    parser.set_defaults(rebuild=False)
    parsed_args = parser.parse_args()


if __name__ == '__main__':
    load_options()
    train()