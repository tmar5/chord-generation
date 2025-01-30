# INF8225 - Artificial Intelligence: Probabilistic and Learning Techniques for Chord Generation in a Music Score

Téo Marcin, Mathias Gonin, Victor Petit, Julien Hage  
Polytechnique Montréal  
\{teo.marcin, mathias.gonin, victor.petit, julien.hage\}@polymtl.ca

#### Abstract

Our work focused on generating chords from a melody using Machine Learning. We use MusicXML files as data, a format based on XML that allows the manipulation of music scores. This problem is underexplored in Artificial Intelligence, so we drew inspiration from Natural Language Processing (NLP) models to capture the temporal dependencies between the notes as well as between the chords played in each measure. Our results were evaluated using a test set, as well as by listening to the audio generated with our scripts and the MuseScore software.


## 1 Introduction

Music generation is an increasingly popular topic in Machine Learning research. Recent technologies like GANs (Generative Adversarial Networks) have led to very interesting results such as MuseGan [Dong et al. (2017) Dong, Hsiao, Yang, and Yang] or Google’s SynthGAN [Engel et al. (2019) Engel, Krishna Agrawal, Chen, Gulrajani, Donahue, and Roberts]. 

While some articles focus on melody generation, we are interested in generating chords from melodies. Chords are essential to accompany a melody, but they can represent challenges for novice musicians, especially those with limited knowledge of music theory. Therefore, some articles have tackled this topic and attempted a measure-based approach, where a chord is generated for a measure using only the notes in that measure. 

The article [Chen et al. (2015) Chen, Qi, and Zhou] proposes applying several basic and advanced models to this task: logistic regression, naive Bayes networks, SVM, Random Forests, Boosting, and HMM. Radial SVM, Random Forest, and Boosting give the best results for chord prediction on a single measure, with accuracy exceeding 60%, but restricting to 7 chord classes. 

However, music is sequential information, and some authors have used temporal dependencies between the notes and developed a sequential model. In the article [Lim et al. (2017) Lim, Rhyu, and Lee], for example, chord generation from a melody is done via a BiLSTM network. A TimeDistributed layer is applied to each measure, and the authors also obtain the best results (accuracy of 0.5) with a sequence of 4 measures.

We wanted to start from this article to obtain a first Seq2Vec model and thus implemented a BiLSTM-type architecture. However, in musical composition, musicians typically choose the chord for a measure based on the notes present in the measure and the chords used in the previous measures. In fact, two measures can have exactly the same notes in the same order - or even no notes at all (rests) - and correspond to different chords. 

We then decided to use the dependencies between measures, which themselves are sequences of notes, inspired by the Seq2Seq model widely used in NLP [Sutskever et al. (2014) Sutskever, Vinyals, and Le]. 

The results were measured on a test set, and our second Seq2Seq approach proved to be more accurate than the BiLSTM approach. But accuracy does not seem to be a sufficient measure of the coherence of music. Indeed, it depends on the correctness — i.e., whether the chord is within the key of the melody — on the diversity of the chords (as a piece with the same chord played repeatedly will feel emotionally empty), and on several other measures that are difficult to describe.

We therefore created a script that generates the chords in audio played on guitar while also playing the melody in parallel, so that we could judge the generated progression by ear. 

Thus, our tests also have an auditory component, which, although subjective, seems just as important. 

This report aims to explain our approach and results: we will first cover the extraction and processing of data, then detail the architecture of the models implemented. Finally, we will analyze the results and discuss our generation of audio files before concluding with potential improvements to our work.


## 2 Data

We have 2236 music scores from the Wikifonia database. These data are in the form of MusicXML (.mxl) files. They are classical music scores, with accompanying chords to the melody. They can be opened and edited with software like MuseScore. Most of these pieces are Western pop music, but there is also jazz, classical, and some world music, such as Chinese music. Some .mxl files are available as examples in the code link.

### 2.1 Data Extraction

To extract the data, we converted the MXL files to XML, then from XML to CSV. The MXL format is actually a zip archive containing a METAINF file (which includes a container.xml) and an .xml file containing the desired information. This information was then extracted using Python's "xml" library and converted to a CSV file by selecting the relevant details, such as measure numbers, pitch, octaves, and durations of the notes, chord pitches and modes, and key information.

### 2.2 Data Preprocessing

To make the information as simple and useful as possible, we used `key_fifths` to determine the key of the piece. To standardize the data, we transposed all songs into the key of C Major. We kept the measure number, the pitch and duration of the notes, and the chord pitches. We retained one chord per measure. To simplify the data, we segmented the chord modes into two distinct modes: major — associated with a joyful sound — and minor — associated with a darker sound. These are the two main modes in music. Thus, a G#m7dim, for example, would be transformed into G#:min (min for minor). A rest corresponds to the absence of a note.

Thus, there are 12 classes for a note (12 pitches: A, A#, B, C, ...). We convert a note into a OneHot vector. We then created Numpy tensors (npy), where the first dimension corresponds to the number of groups of n measures (explained below), the second dimension corresponds to a measure, and the third dimension corresponds to the note. Padding was applied to ensure that note sequences (measures) have the same size (size 32). These are tensors of size Number of songs x Number of measures x Note vector size. Then, two types of data are used to train two types of models. The BiLSTM will take as input a sequence of notes belonging to a single measure, while the Seq2Seq model will take a sequence of notes belonging to multiple measures. We use overlapping sequences of measures from the same song, for example, the sequence of measures [1, 2, 3, 4], then the sequence [2, 3, 4, 5], with the only condition being that the measures must come from the same song. However, the test set will only use non-overlapping sequences of measures, such as [1, 2, 3, 4] followed by [5, 6, 7, 8], to obtain results comparable to those in the article [Lim et al. (2017) Lim, Rhyu, and Lee].

After testing our models, we decided to add the note duration to our model. This feature seems very important because a whole note (which lasts 4 beats) should not have the same weight as an eighth note (which lasts half a beat) on the harmony. We chose to do this in two different ways. First, by adding a 13th column to the note vector corresponding to the normalized duration of the note. Second, by repeating the note in the measure (which is a sequence of notes) a number of times corresponding to its duration. For example, a half note (2 beats) will be the same note (same vector) repeated 16 times, because a measure is a sequence of 32 notes, and the half note represents half a measure. The results obtained are better when we take duration into account, and similar for both methods, although slightly better for the repeated note sequence. We will therefore choose this second method in the following.

### 2.3 Division into Training, Validation, and Test Sets

The data were divided into training, test, and validation sets. The training set includes 1788 songs, and the test set includes 458. We use 10% of the training set (179 songs) to create a validation set necessary for fine-tuning the learning process and avoiding overfitting, as well as selecting the right hyperparameters.


## 3 Architectures and Methods Used

The chosen architectures were implemented in Python using the PyTorch library, allowing for GPU integration to speed up learning. We also used Google Colab to leverage more powerful graphic processors.

### 3.1 BiLSTM Recurrent Neural Network

The first method considered was a BiLSTM architecture.

This model takes into account the dependencies of the notes within a single measure. Indeed, a chord is generated for each measure and is primarily related to the notes that make up the measure. In this approach, the model's inputs are constituted by the sequence of notes in each measure. The training and validation measures are separated from each other, and we no longer look at their relationships within the same song.

The architecture that provided the best results is a model with two bidirectional LSTM layers, with hidden dimensions of 256, a dropout of 0.3 between each layer, a linear layer reducing the output dimensions to 24, and a log_softmax layer. The loss function is the negative log-likelihood, which, combined with the log_softmax layer, allows us to use cross-entropy.

The notes are sequentially passed into the network. Thus, the 32 notes of a measure are fed into the network, each note being a one-hot vector corresponding to the played note, with a dimension of 12, sometimes with added features (as explained in part 3). The output is a chord corresponding to the measure, a one-hot vector of dimension 24.

For learning, we used ADAM, with a learning rate of 0.001, along with a scheduler to decrease it when the loss on the training set stagnates. Around forty epochs were sufficient for the network to learn. The training is done by monitoring the loss on the validation set, and the selected model is the one corresponding to the lowest loss on this set.

### 3.2 Seq2Seq Encoder-Decoder Model

A Seq2Seq model allows us to take into account not only the dependencies of each chord with respect to the notes that make up its measure, but also the dependency of each chord on the chords of neighboring measures. A chord is, in fact, the result of harmony relative to the preceding and following chords, in addition to the obvious dependencies with the notes of the measure it is associated with.

For the network input, we used sequences of notes from multiple consecutive measures, with the trivial condition that the measures in the sequence must all belong to the same song. The best results were obtained for a sequence of 4 measures, i.e., 32 * 4 notes. Indeed, the model struggles to retain temporal dependencies for longer sequences of measures. Additionally, a chord is mainly related to the measures that are closest to it.

The Seq2Seq model is of the encoder-decoder type. The encoder is unidirectional. Tests showed that using a bidirectional encoder and initializing each layer of the decoder with the concatenation of the last hidden/state cell for both directions yields very similar performance to using a unidirectional encoder. The encoder consists of two LSTM layers with hidden dimensions of 256, and a dropout of 0.3 between these layers. The decoder retrieves the last hidden/state cell for each layer to initialize. Similarly, it is built from two LSTM layers with hidden dimensions of 256, dropout of 0.3, followed by a linear layer that reduces the generated chord dimensions to 24, and then a log_softmax layer.

For training this model, we use the negative log-likelihood, which, combined with the log_softmax layer of the model, is equivalent to the cross-entropy loss function. We use gradient clipping, which sets the maximum gradient value to 1.0 (to prevent the exploding gradient problem), and the ADAM optimizer. Training was done with the help of teacher-forcing. With a fixed probability of 0.5, the next input to the decoder is the true chord that should have been predicted in the previous output. Otherwise, the input to the decoder is the previous prediction. It is also important to note that to generate a sequence of chords, the decoder is initialized with a `<start>` token (a one-hot vector of dimension 25, with 1 at the last position). This prevents learning the identity by shifting the decoder inputs and targets by one position. Furthermore, it also allows us not to use targets (i.e., the first chord in the sequence to be generated) during the model's test phase. Thus, the model's outputs are vectors of dimension 25, with 24 chord classes and 1 class corresponding to the start token. Training this model is longer and requires around a hundred epochs.

### 3.3 Seq2Seq Encoder-Decoder Model Coupled with an Autoencoder

![](https://cdn.mathpix.com/cropped/2025_01_30_1e5b80436efff83e4364g-3.jpg?height=346&width=676&top_left_y=321&top_left_x=1178)

**Autoencoder Architecture**  
To improve our solution, we thought it might be interesting to compress our data before feeding it into our Seq2Seq model. Previously, each group of 4 measures was represented by a matrix of size (32 * 4, 12) consisting of one-hot vectors. We could imagine that learning a dense representation of our data would make the learning process more efficient. An autoencoder learns a dense representation of the data by having the same data as input and output, with a bottleneck in the middle, from which it tries to reconstruct the original data. This structure could therefore compress each measure, with the idea being to feed into our Seq2Seq architecture a group of 4 measures that is no longer represented by a matrix of size (32 * 4, 12) but instead a matrix of size (4, dim_compressed), where `dim_compressed` corresponds to the size of the vector representing a measure after encoding.

Since we want to compress the data for each measure, the input dataset here does not correspond to notes grouped by 4 measures (as in the Seq2Seq architecture), but instead notes grouped by individual measures. Thus, the measures are represented by matrices of size (32, 12), which we flatten into a large vector before passing it into the autoencoder. The goal of the autoencoder is to reduce the dimension of this vector to the size `dim_compressed`.

Since the autoencoder is a specific case of an encoder-decoder, our Seq2Seq implementation was adapted to create the autoencoder architecture. It consists of an encoder — an LSTM followed by a linear layer, with the output dimension corresponding to `dim_compressed` — and a decoder — an LSTM followed by a linear layer — that predicts each note of the encoded measure.

As with the Seq2Seq model, it is necessary to initialize the decoder with a start token to prevent overfitting and also avoid using the first note of each measure, which is the target of the model, during the test phase. Our notes, which form the model's targets, are actually represented by one-hot vectors of length 13, where the first 12 elements correspond to the 12 note classes, and the 13th element corresponds to the start token.

For evaluation, to prevent placing too much importance on the zero vectors added to the measure matrices due to padding (done to make all measures have a size of (32, 12)), we calculate an accuracy where we do not consider the autoencoder's predictions on these zero vectors.

We obtained good results for data compression by encoding each measure into a vector of dimension 50, with a teacher_forcing_ratio of 0.3.

Unfortunately, when we tried to feed these encoded measures into our Seq2Seq model, the training of the network did not work at all. We didn’t have enough time to figure out what went wrong, so we were unable to integrate the autoencoder into our pipeline.


## 4 Results

We denote D1 as the dataset with the temporal feature repeated the number of times corresponding to its duration.

Di with i different from 1: set of i measures obtained by the same treatment as D1, overlapping (but no overlap for the test set as explained in section 2.2).

### 4.1 Model Accuracy

The results obtained are presented in Table 1 below. They were obtained with a batch size of 2048.

| Model | Data | Accuracy <br> validation | Accuracy <br> test |
| :---: | :---: | :---: | :---: |
| BiLSTM | D1 | 0.523 | 0.452 |
| Seq2Seq | D2 | 0.548 | 0.451 |
| Seq2Seq | D4 | 0.550 | 0.471 |
| Seq2Seq | D6 | 0.516 | 0.460 |
| Seq2Seq | D8 | 0.604 | 0.443 |

**Table 1: Results**  
![](https://cdn.mathpix.com/cropped/2025_01_30_1e5b80436efff83e4364g-4.jpg?height=612&width=852&top_left_y=1556&top_left_x=182)

**Figure 1: Loss value evolution for the Seq2Seq model, D4, during training**

The autoencoder provides good results when trained for 50 epochs, with an accuracy (where zero vectors from padding are ignored) on test inputs of 0.948 and the following learning curves shown in Figures 3 and 4. Learning a denser form of the data is effective.

Unfortunately, we were not able to observe the performance of this encoding of the data fed into the Seq2Seq model. Our attempts failed to get the model to learn from this encoded data, so we could not integrate this autoencoder into our pipeline. We can therefore question whether the learned representation for each measure is a relevant and usable representation of the data.  
![](https://cdn.mathpix.com/cropped/2025_01_30_1e5b80436efff83e4364g-4.jpg?height=478&width=667&top_left_y=658&top_left_x=1157)

**Figure 2: Evolution of the loss value during the training of the autoencoder**  
![](https://cdn.mathpix.com/cropped/2025_01_30_1e5b80436efff83e4364g-4.jpg?height=481&width=678&top_left_y=1350&top_left_x=1149)

**Figure 3: Evolution of accuracy during the training of the autoencoder**

## 4.2 Listening to the Results

As explained in the introduction, the precision measurement alone is not sufficient to determine whether the results are correct. To measure the coherence of the results obtained, we generated audio files playing both the melody and the guitar chords at each measure. For this, we created a Python script that adds a second voice (played on the guitar) to the original mxl file. This second voice plays the chords generated at each measure using a dictionary of basic music theory (for minor/major chords). See Figure 4.  
![](https://cdn.mathpix.com/cropped/2025_01_30_1e5b80436efff83e4364g-5.jpg?height=289&width=801&top_left_y=181&top_left_x=205)

**Figure 4: Generated sheet music. The top line corresponds to the melody, the bottom line to the guitar harmony. The chords are marked in black.**

Once the .mxl files are generated, MuseScore software is used to produce a .wav audio file. Resulting and ground truth audio files are provided in the code link as an example.

To measure the coherence of the chord generation, we had 4 different people listen to the audios with real chords and the generated chords. Their task was to determine whether the heard chords matched the original ones or the generated ones. Unfortunately, due to a small number of chords sounding wrong, all 4 subjects correctly identified the generated chords. However, they all emphasized the coherence of the generated progression. Although this test was not conclusive, it remains important to measure the relevance of our results. A larger-scale test would have been a good indicator of coherence and could have helped improve our model.

## 5 Conclusion

In conclusion, the best results were obtained with a Seq2Seq encoder-decoder model. The bidirectionality of the encoder in this model, or the BiLSTM model, does not bring any clear improvement and requires much longer computation times. A chord seems to be more closely linked to the chords that follow it, but less to those that precede it.

The plateau of accuracy achieved by our different models may be related to the intrinsic structure of the data. Indeed, the music used comes from very varied genres. Jazz chords are not governed by the same rules as Chinese or pop music chords. The model may therefore have difficulties generalizing its learning. We observed a decrease in the loss value on the training set down to very low values (overfitting to values around 0.1). However, the loss on the validation set decreases much less and much slower. We could improve our results by using more data for training, for example through cross-validation, and especially by targeting this data, for instance, restricting ourselves to pop music.

Additionally, it may be useful to add a batch normalization layer to speed up training, which remains very slow in our various trials. We should also explore new relevant features, such as octaves. We could explore new dimensionality reduction techniques, using our autoencoder, or even adding an embedding layer in the Seq2Seq model.

Finally, we segmented the chords into only two different modes. We could add modes such as seventh chords, which add tension to the chord progression, to obtain more complex, diverse, and musically coherent progressions.


## References

[Chen et al.(2015)Chen, Qi, and Zhou] Ziheng Chen, Jie Qi, and Yifei Zhou. 2015. Machine Learning in Automatic Music Chords Generation.

[Dong et al.(2017)Dong, Hsiao, Yang, and Yang] Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, and Yi-Hsuan Yang. 2017. http://arxiv.org/abs/1709.06298 MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment. arXiv e-prints, page arXiv:1709.06298.

[Engel et al.(2019)] Engel, Krishna Agrawal, Chen, Gulrajani, Donahue Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, and Adam Roberts. 2019. http://arxiv.org/abs/1902.08710 GANSynth: Adversarial Neural Audio Synthesis. arXiv e-prints, page arXiv:1902.08710.

[Lim et al.(2017)Lim, Rhyu, and Lee] Hyungui Lim, Seungyeon Rhyu, and Kyogu Lee. 2017. http://arxiv.org/abs/1712.01011 Chord Generation from Symbolic Melody Using BLSTM Networks. arXiv e-prints, page arXiv:1712.01011.

[Sutskever et al.(2014)Sutskever, Vinyals, and Le] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. http://arxiv.org/abs/1409.3215 Sequence to Sequence Learning with Neural Networks. arXiv e-prints, page arXiv:1409.3215.

