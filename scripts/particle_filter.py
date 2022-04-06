import numpy as np

import matplotlib.pyplot as plt

# state spaces
from madmom.features import RNNDownBeatProcessor
from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel
from madmom.ml.hmm import TransitionModel, ObservationModel
from scipy.io import wavfile


class meter():
    def __init__(self, st, tm, tm1, om):
        self.st = st
        self.tm = tm
        self.tm = tm1
        self.om = om


class BObservationModel(ObservationModel):
    """
    Observation model for beat tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BeatStateSpace` instance
        BeatStateSpace instance.
    observation_lambda : int
        Split one beat period into `observation_lambda` parts, the first
        representing beat states and the remaining non-beat states.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    """

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        # compute observation pointers
        # always point to the non-beat densities
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        # unless they are in the beat range of the state space
        border = 1. / observation_lambda
        pointers[state_space.state_positions < border] = 1
        # instantiate a ObservationModel with the pointers
        super(BObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, )
            Observations (i.e. 1D beat activations of the RNN).

        Returns
        -------
        numpy array, shape (N, 2)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats and beats.

        """
        # init densities
        log_densities = np.empty((len(observations), 2), dtype=np.float)
        # Note: it's faster to call np.log 2 times instead of once on the
        #       whole 2d array
        log_densities[:, 0] = np.log((1. - observations) /
                                     (self.observation_lambda - 1))
        log_densities[:, 1] = np.log(observations)
        # return the densities
        return log_densities


class BDObservationModel(ObservationModel):
    """
    Observation model for beat and downbeat tracking with particle filtering.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : int
        Split each (down-)beat period into `observation_lambda` parts, the
        first representing (down-)beat states and the remaining non-beat
        states.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    """

    def __init__(self, state_space, observation_lambda):

        if observation_lambda[0] == 'B':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            border = 1. / observation_lambda
            pointers[state_space.state_positions % 1 < border] = 1
            # the downbeat (i.e. the first beat range) points to density column 2
            pointers[state_space.state_positions < border] = 2
            # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'N':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            for i in range(observation_lambda):
                border = np.asarray(state_space.first_states) + i
                pointers[border[1:]] = 1
                # the downbeat (i.e. the first beat range) points to density column 2
                pointers[border[0]] = 2
                # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'G':
            observation_lambda = float(observation_lambda[1:])
            pointers = np.zeros((state_space.num_beats + 1, state_space.num_states))
            for i in range(state_space.num_beats + 1):
                pointers[i] = gaussian(state_space.state_positions, i, observation_lambda)
            pointers[0] = pointers[0] + pointers[-1]
            pointers[1] = np.sum(pointers[1:-1], axis=0)
            pointers = pointers[:2]
            super(BDObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, 2)
            Observations (i.e. 2D activations of a RNN, the columns represent
            'beat' and 'downbeat' probabilities)

        Returns
        -------
        numpy array, shape (N, 3)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats, beats and
            downbeats.

        """
        # init densities
        log_densities = np.empty((len(observations), 3), dtype=np.float)
        # Note: it's faster to call np.log multiple times instead of once on
        #       the whole 2d array
        log_densities[:, 0] = np.log((1. - np.sum(observations, axis=1)) /
                                     (self.observation_lambda - 1))
        log_densities[:, 1] = np.log(observations[:, 0])
        log_densities[:, 2] = np.log(observations[:, 1])
        # return the densities
        return log_densities

    # resampling methods


def resample(particles, weights, method, state_space):
    if method == "universal":
        assert state_space is not None
        new_particles = []
        J = len(particles)
        weights = weights / sum(weights)
        r = np.random.uniform(0, 1 / J)
        i = 0
        c = weights[0]
        for j in range(J):
            U = r + j * (1 / J)
            while U > c:
                i += 1
                c += weights[i]
            new_particles = np.append(new_particles, particles[i])
        new_particles = new_particles.astype(int)
        median = int(np.median(new_particles))
        # if np.in1d(state_space.first_states,median).any():
        #     for i in range(state_space.num_beats):
        #         if np.in1d(state_space.first_states[i],median).any():
        #             if state_space.state_intervals[0]<state_space.state_intervals[median]*2<state_space.state_intervals[-1]:
        #                 nn = np.random.choice(len(particles), 1)
        #                 new_particles[nn]=state_space.first_states[i][state_space.state_intervals[state_space.first_states[i]] == state_space.state_intervals[median] * 2 ]         #investigates half tempo
        #             if state_space.state_intervals[0] < state_space.state_intervals[median] * 0.5 < state_space.state_intervals[-1]:
        #                 nn = np.random.choice(len(particles), 1)
        #                 new_particles[nn] = state_space.first_states[i][state_space.state_intervals[state_space.first_states[i]] == int(state_space.state_intervals[median] * 0.5)]   #investigates double tempo
        return new_particles

    elif method == "systematic":
        N = len(weights)
        # make N subdivisions, choose positions
        # with a consistent random offset
        positions = (np.random.randint(0, N) + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N & j < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return particles[indexes]

    elif method == "stratified":
        N = len(weights)
        # make N subdivisions, chose a random position within each one
        positions = (np.random.randint(0, N) + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N & j < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return particles[indexes]
    else:
        raise NameError("there is no such method name")


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)  # /(np.sqrt(2.*np.pi)*sig)


def densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if state_model.num_beats > 1:  # beat and downbeat
        if len(np.shape(observation_model.pointers)) == 1:     #   B or N
            new_obs[np.argwhere(observation_model.pointers == 2)] = observations[
                1]  # * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)]
            new_obs[np.argwhere(observation_model.pointers == 1)] = observations[
                0]  # * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 1)]
            new_obs[np.argwhere(
                observation_model.pointers == 0)] = 0.03  # ((1-alpha) * densities[2] * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)])
        else:    #  G
            new_obs = observation_model.pointers[0] * observations[1] + observation_model.pointers[1] * observations[
                0]  # observation_model.pointers[0] = downbeat weigths   observation_model.pointers[1] = beat weigths
            new_obs[new_obs < 0.005] = 0.03
    else:  # only beat
        if len(np.shape(observation_model.pointers)) != 2:  # B or N
            new_obs[np.argwhere(
                observation_model.pointers == 2)] = observations  # * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)]
            new_obs[np.argwhere(
                observation_model.pointers == 0)] = 0.03  # ((1-alpha) * densities[2] * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)])
        elif len(np.shape(observation_model.pointers)) == 2:  # G
            new_obs = observation_model.pointers[
                          0] * observations  # observation_model.pointers[0] = downbeat weigths   observation_model.pointers[1] = beat weigths
            new_obs[new_obs < 0.005] = 0.03

    # return the densities
    return new_obs


class particle_filter:

    np.random.seed(1)
    PARTICLE_SIZE = 1000
    STATE_FLAG = 0
    MIN_BPM = 55.
    MAX_BPM = 260.  # 260   215.    370.
    NUM_TEMPI = 300
    TRANSITION_LAMBDA = 5
    OBSERVATION_LAMBDA = "B60"  # "N3"   "G0.0025" "B20"  "G0.0025"
    THRESHOLD = None
    fps = 50
    T = 1 / fps

    # BEATS_PER_BAR = [3, 4]

    def __init__(self, beats_per_bar, particle_size=PARTICLE_SIZE, state_flag=STATE_FLAG, min_bpm=MIN_BPM,
                 max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD,
                 fps=None, **kwargs):
        self.particle_size = particle_size
        self.particle_filter = []
        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.fps = fps
        self.observation_lambda = observation_lambda
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        # make sure the other arguments are long enough by repeating them
        # TODO: check if they are of length 1?
        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda,
                                          len(beats_per_bar))
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(beats_per_bar) == len(transition_lambda)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi`, `num_beats` '
                             'and `transition_lambda` must all have the same '
                             'length.')
        # get num_threads from kwargs
        num_threads = min(len(beats_per_bar), kwargs.get('num_threads', 1))
        # init a pool of workers (if needed)
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map
        # convert timing information to construct a beat state space
        if state_flag == 0:
            min_interval = 60. * fps / max_bpm
            max_interval = 60. * fps / min_bpm
        elif state_flag == 1:
            min_interval = 4 * 60. * fps / max_bpm
            max_interval = 4 * 60. * fps / min_bpm
        # model the different bar lengths

        self.meters = []
        for b, beats in enumerate(self.beats_per_bar):
            st = BarStateSpace(beats, min_interval[b], max_interval[b],
                               num_tempi[b])
            tm = BarTransitionModel(st, transition_lambda[b])
            tm1 = list(TransitionModel.make_dense(tm.states, tm.pointers, tm.probabilities))
            om = BDObservationModel(st, observation_lambda)
            st.last_states = list(np.concatenate(st.last_states).flat)
            self.meters.append(meter(st, tm, tm1, om))
            # self.particle_filter.append(HiddenMarkovModel(tm, om))
        pass
        # save variables

    def process(self, activations, **kwargs):

        """
        Detect the (down-)beats in the given activation function.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        # pylint: disable=arguments-differ
        import itertools as it
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))

        T = 1 / self.fps

        for counter, meter in enumerate(self.meters):
            counter = 0
            # border = 1. / meter.st.state_intervals[-1]
            # stateee = np.arange(meter.st.num_states)

            # fig = plt.figure(figsize=(1800 / 96, 800 / 96), dpi=96)
            # line1 = plt.scatter(meter.st.state_positions, np.max(meter.st.state_intervals) - meter.st.state_intervals,
            #                     marker='o', color='grey', alpha=0.2)

            # plt.show()
            # plt.show()
            #
            #
            # for i in range(len(meter.st.first_states[0])-1):
            #     for b in range(meter.st.num_beats):
            #         line1 = plt.scatter(meter.st.state_positions[stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i + 1]]],
            #             (len(meter.st.first_states[b]) - i - 1) * np.ones(
            #                 len(stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i + 1]])), marker='o', color='grey', alpha=0.2)

            # line1, = plt.plot(
            #     meter.st.state_positions[stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i + 1]]],
            #     (len(meter.st.first_states[b]) - i - 1) * np.ones(
            #         len(stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i + 1]])), 'o',
            #     color='grey')

            path = np.zeros((1, 2), dtype=float)
            position = []
            particles = np.sort(np.random.randint(0, meter.st.num_states - 1, self.particle_size))
            downbeat = meter.st.first_states[0]
            beat = np.squeeze(meter.st.first_states[1:])

            # activations[activations < 0.4] =  0.03  # 0.0005
            # for i in range(len(activations)-3):
            #     if activations[i+1][0]<activations[i][0]:
            #         activations[i + 1][0]= 0.03
            #     if activations[i+2][0]<activations[i][0]:
            #         activations[i + 2][0]= 0.03
            #     if activations[i+3][0]<activations[i][0]:
            #         activations[i + 3][0]= 0.03
            #     if activations[i+1][1]<activations[i][1]:
            #         activations[i + 1][1]= 0.03
            #     if activations[i+2][1]<activations[i][1]:
            #         activations[i + 2][1]= 0.03
            #     if activations[i+3][1]<activations[i][1]:
            #         activations[i + 3][1]= 0.03

            # activations = activations[250:]



            for i in range(len(activations)):
                counter += 1

                position = np.median(meter.st.state_positions[particles])
                gathering = int(np.median(particles))

                #
                # if abs(gathering - downbeat) < 3 and (counter * T) - path[-1][0] > 0.27:
                #     path = np.append(path, [[counter * T, 1]], axis=0)
                # elif abs(gathering - beat[meter.st.state_intervals[beat] == meter.st.state_intervals[gathering]]).any() < 3 and (counter * T) - path[-1][0] > 0.27:
                #     path = np.append(path, [[counter * T, 2]], axis=0)

                if abs(gathering - downbeat[
                    meter.st.state_intervals[downbeat] == meter.st.state_intervals[gathering]]) < int(.07 / T) + 2 and (
                        counter * T) - path[-1][0] > .6 * T * meter.st.state_intervals[gathering]:
                    path = np.append(path, [[counter * T, 1]], axis=0)
                elif (abs(gathering - beat[meter.st.state_intervals[beat] == meter.st.state_intervals[gathering]]) < (
                        int(.07 / T)) + 2).any() and (counter * T) - path[-1][0] > .6 * T * meter.st.state_intervals[
                    gathering]:
                    path = np.append(path, [[counter * T, 2]], axis=0)


                #
                # if (abs(gathering - downbeat) < int(.07 / T)).any() and (
                #         counter * T) - path[-1][0] > .5 * T * meter.st.state_intervals[gathering]:
                #     path = np.append(path, [[counter * T, 1]], axis=0)
                # elif (abs(gathering - beat) < int(
                #         .07 / T)).any() and (counter * T) - path[-1][0] > .5 * T * meter.st.state_intervals[
                #     gathering]:
                #     path = np.append(path, [[counter * T, 2]], axis=0)

                # if position < 1. / 15 and (counter * T) - path[-1][0] > 0.27:
                #     path = np.append(path, [[counter * T, 1]], axis=0)
                # elif position % 1 < 1. / 15 and (counter * T) - path[-1][0] > 0.27:
                #     path = np.append(path, [[counter * T, 2]], axis=0)

                # position = meter.st.state_positions[np.argmax(np.bincount(particles))]
                obs = densities(activations[i], meter.om, meter.st)



                # plot presentation
                if counter % 1 == 1:  # counter > 0:  #
                    # plt.figure(figsize=(1800 / 96, 800 / 96), dpi=96)
                    axes = plt.gca()
                    axes.set_xlim(-0.05, 4.05)
                    axes.set_ylim(-2, 80)   # (-2 , 40)
                    m = np.r_[True, particles[:-1] != particles[1:], True]
                    counts = np.diff(np.flatnonzero(m))
                    unq = particles[m[:-1]]
                    part = np.c_[unq, counts]

                    plt.scatter(meter.st.state_positions, np.max(meter.st.state_intervals) - meter.st.state_intervals,
                                marker='o',
                                color='grey', alpha=0.08)
                    plt.scatter(meter.st.state_positions[meter.om.pointers == 2],
                                np.max(meter.st.state_intervals) - meter.st.state_intervals[meter.om.pointers == 2],
                                marker='o',
                                color='green', alpha=activations[i][1])
                    plt.scatter(meter.st.state_positions[meter.om.pointers == 1],
                                np.max(meter.st.state_intervals) - meter.st.state_intervals[meter.om.pointers == 1],
                                marker='o',
                                color='yellow', alpha=activations[i][0])
                    # for m in range (len(meter.st.state_positions)):
                    #     plt.scatter(meter.st.state_positions[m], 55 - meter.st.state_intervals[m], marker='o',
                    #             color='yellow', alpha=obs[m])
                    plt.scatter(meter.st.state_positions[part[:, 0]],
                                np.max(meter.st.state_intervals) - meter.st.state_intervals[part[:, 0]],
                                marker='o',
                                s=part[:, 1], color="red")
                    plt.axvline(x=position)
                    plt.pause(0.001)
                    fig.clear()

                plt.draw()


                # obs = densities(activations[i], meter.om, meter.st)
                particles = resample(particles=particles, weights=obs[particles], method="universal",
                                     state_space=meter.st)
                last = particles[np.in1d(particles, meter.st.last_states)]
                state = particles[~np.in1d(particles, meter.st.last_states)] + 1
                for j in range(len(last)):
                    args = np.argwhere(meter.tm[1] == last[j])
                    probs = meter.tm[2][args]
                    nn = np.random.choice(np.squeeze(meter.tm[0][args]), 1, p=(np.squeeze(probs)))
                    state = np.append(state, nn)
                particles = state

            # plt.draw()
            # stateee = np.arange(meter.st.num_states)
            # for i in range(len(meter.st.first_states[0])):
            #     for b in range(meter.st.num_beats):
            #         plt.plot(meter.st.state_positions[
            #                      stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i + 1]]],
            #                  (len(meter.st.first_states[0]) - i + 1) * np.ones(
            #                      len(stateee[meter.st.first_states[0][i]:meter.st.first_states[0][i + 1]])), 'o',
            #                  color='grey')
            # plt.show()

            # for i in range(len(meter.st.first_states[0])):
            #     for b in range(meter.st.num_beats):
            #         plt.plot(meter.st.state_positions[
            #                      stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i + 1]]],
            #                  (len(meter.st.first_states[0]) - i + 1) * np.ones(
            #                      len(stateee[meter.st.first_states[0][i]:meter.st.first_states[0][i + 1]])), 'o',
            #                  color='grey')
            # plt.show()
        path = path[1:]
        beats = path[:, 0][path[:, 1] == 2]
        downs = path[:, 0][path[:, 1] == 1]
        return beats, downs
        # stateee = np.arange(meter.st.num_states)
        # for i in range(len(meter.st.first_states[0])):
        #     for b in range(meter.st.num_beats):
        #       plt.plot(meter.st.state_positions[stateee[meter.st.first_states[b][i]:meter.st.first_states[b][i+1]]], (len(meter.st.first_states[0])-i+1) * np.ones(len(stateee[meter.st.first_states[0][i]:meter.st.first_states[0][i+1]])), 'o', color='grey')
        # plt.show()
        # for i in range(len(meter.st.first_states[0])):
        #     plt.plot(particles[meter.st.first_states[0][i] < particles < meter.st.first_states[0][i+1]], i * np.ones(len(particles)), 'o')
        #     plt.hold
        #
        # plt.show()


# RNNDownBeat = RNNDownBeatProcessor()
# fs, data3 = wavfile.read("D:\datasets\GTZAN/audio/blues/blues.00000.wav")
# activation = RNNDownBeat(data3)
# particle = particle_filter(beats_per_bar=[4], fps=50)
# beats, downs = particle.process(activation)
# pass
