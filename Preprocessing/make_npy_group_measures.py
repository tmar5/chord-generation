"""
Convert the dataset into correct inputs/targets
"""
import csv
import glob
import numpy as np
from keras_preprocessing import sequence

note_dictionary = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

chord_dictionary = ['C:maj', 'C:min',
                    'C#:maj', 'C#:min',
                    'D:maj', 'D:min',
                    'D#:maj', 'D#:min',
                    'E:maj', 'E:min',
                    'F:maj', 'F:min',
                    'F#:maj', 'F#:min',
                    'G:maj', 'G:min',
                    'G#:maj', 'G#:min',
                    'A:maj', 'A:min',
                    'A#:maj', 'A#:min',
                    'B:maj', 'B:min']


def one_hot_encoding(length, one_index):
    """Return the one hot vector."""
    vectors = np.zeros(length)
    vectors[one_index] = 1
    return vectors


def make_npy(_input, nb_measure=4, padding_measure=32):
    """
    For Seq2Seq model:
        - input = group of "nb_measure" measures, measures are related to only one song
        - inputs have their position reversed for decoder !
        - each note is repeated number of time proportionnaly to its duration
        - target: each group of measure begin with <start> token

    Train set : each measure is used several time [1,2,3,4], [2,3,4,5] and not [1,2,3,4], [5,6,7,8]
    Test set : each measure is used one time [1,2,3,4], [5,6,7,8]

    decoder:
        - start token: <start> => output target = chord i ...
        - no <end> token because we know the length of the seq to generate,
        no need to wait for the decoder to output <end>

    For BILSTM: -set nb_measure to 1
                -remove additional start chord
        -
    :param _input:
    :param nb_measure: nb_measure in each group of measures
    :param padding_measure: nb notes in one measure
    :return:
    """
    np.set_printoptions(threshold=np.inf)

    if _input == 'train':
        file_path = 'dataset/new_train/*.csv'
    elif _input == 'test':
        file_path = 'dataset/new_test/*.csv'
    else:
        raise Exception("Input error")
    csv_files = glob.glob(file_path)

    note_dict_len = len(note_dictionary)
    chord_dict_len = len(chord_dictionary) + 1  # for start token: one more class

    measure_length = 32  # as a array
    measure_duration = 16  # On CSV file

    # list for final input/target vector. Each element: sequence of processed measures for one song
    result_input_matrix = []
    result_target_matrix = []

    # make the matrix from csv data
    for csv_path in csv_files:
            
        print(csv_path)
        csv_ins = open(csv_path, 'r', encoding='utf-8')
        next(csv_ins)  # skip first line
        reader = csv.reader(csv_ins)

        # list for each song(each npy file) in the test set
        note_sequence, song_sequence, chord_sequence = list(), list(), list()
        pre_measure = None
        for line in reader:
            try:
                measure = int(line[0])
                chord = line[1]
                note = line[2]
                duration = float(line[3])
    
                # find one hot index
                chord_index = chord_dictionary.index(chord)
                note_index = note_dictionary.index(note)
    
                one_hot_note_vec = one_hot_encoding(note_dict_len, note_index)
                one_hot_chord_vec = one_hot_encoding(chord_dict_len, chord_index)
    
                if pre_measure is None:  # case : first line
                    for i in range(int(duration * measure_length / measure_duration)):
                        note_sequence.append(one_hot_note_vec)
                    chord_sequence.append(one_hot_chord_vec)
    
                elif pre_measure == measure:  # case : same measure note
                    for i in range(int(duration * measure_length / measure_duration)):
                        note_sequence.append(one_hot_note_vec)
    
                else:  # case : next measure note
                    song_sequence.append(note_sequence)
                    note_sequence = []
                    for i in range(int(duration * measure_length / measure_duration)):
                        note_sequence.append(one_hot_note_vec)
                    chord_sequence.append(one_hot_chord_vec)
                pre_measure = measure
            except:
                continue
        song_sequence.append(note_sequence)

        # Padding of each measure to padding_measure. Sequence of measures from the same song
        song_sequence_padded = sequence.pad_sequences(song_sequence, maxlen=padding_measure)  # case : last measure note
        result_input_matrix.append(song_sequence_padded)
        result_target_matrix.append(chord_sequence)

    # Groupe les mesures par groupes de nb_measure non overlapping for test and overlapping for train
    if _input == 'test':
        # Troncature pour les dernières mesures si pas assez pour la taille du groupe voulu dans une chanson.
        # Retourne list : nbr_groups_of_measure_dataset*nbr_note_each_group_measure*size_note
        grouped_measures = list()
        forbidden_list = []
        for i,song in enumerate(result_input_matrix):  # song: nbr_measures_song*padding_measure*len(one_note)
            try:
                grouped_measures.extend(
                    [np.flip(np.reshape(song[x:x + nb_measure], (nb_measure * padding_measure, note_dict_len)), axis=1)
                     for x in range(0, len(song), nb_measure) if x + nb_measure <= len(song)])
            except:
                forbidden_list.append(i)
                continue
        # Convert into array: nbr_groups_measures_total_in_dataset*nbr_note_each_group_measure*size_note
        result_input_matrix = np.array(grouped_measures)

        grouped_chords = list()
        for i_song, song in enumerate(result_target_matrix):  # song : list of onehot chords
            # target = seq of chords
            if i_song not in forbidden_list:
                grouped_chords.extend([np.reshape(song[x:x + nb_measure], (nb_measure, chord_dict_len))
                                   for x in range(0, len(song), nb_measure) if x + nb_measure <= len(song)])

        # Convert into array: nbr_groups_measures_total_in_dataset*nb_measure*size_chord
        result_target_matrix_ = np.array(grouped_chords)
        print(len(forbidden_list))
    if _input == 'train':
        # Retourne list : nbr_groups_of_measure_dataset*nbr_note_each_group_measure*size_note
        grouped_measures = list()
        forbidden_list = []
        for i, song in enumerate(result_input_matrix):  # song: nbr_measures_song*padding_measure*len(one_note)
            try:
                grouped_measures.extend(
                [np.flip(np.reshape(song[i:i + nb_measure], (nb_measure * padding_measure, note_dict_len)), axis=1)
                 for i in range(0, len(song) - nb_measure - 1)])
            except:
                forbidden_list.append(i)
                continue

        # Convert into array: nbr_groups_measures_total_in_dataset*nbr_note_each_group_measure*size_note
        result_input_matrix = np.array(grouped_measures)

        grouped_chords = list()
        for i_song, song in enumerate(result_target_matrix):  # song : list of onehot chords
            # target = seq of chords
            if i_song not in forbidden_list:
                grouped_chords.extend([np.reshape(song[i:i + nb_measure], (nb_measure, chord_dict_len))
                                       for i in range(0, len(song) - nb_measure - 1)])

        # Convert into array: nbr_groups_measures_total_in_dataset*nb_measure*size_chord
        result_target_matrix_ = np.array(grouped_chords)

    # Creating onehot for <start> token, will be insert at last position of onehot chord
    start_chord = np.zeros((chord_dict_len, 1), dtype=int)
    start_chord[-1, :] = 1
    # making repetition for each groups of measure
    start_chord = np.tile(start_chord, result_target_matrix_.shape[0])
    start_chord = np.transpose(start_chord)

    result_target_matrix_ = np.insert(result_target_matrix_, 0, start_chord, axis=1)
    filename = str(nb_measure) + '_' + 'measures.npy'

    if _input == 'train':
        np.save('dataset/train_input_' + filename, np.array(result_input_matrix))
        np.save('dataset/train_target_' + filename, np.array(result_target_matrix_))
    elif _input == 'test':
        print(len(result_input_matrix))
        print(len(result_target_matrix_))
        np.save('dataset/test_input_' + filename, np.array(result_input_matrix))
        np.save('dataset/test_target_' + filename, np.array(result_target_matrix_))


if __name__ == '__main__':
    make_npy('test')
    make_npy('train')
