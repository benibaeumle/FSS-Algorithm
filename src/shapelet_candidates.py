# Author: Benedikt Bäumle <benedikt.baeumle@uni-konstanz.de>
#
# License: MIT

import numpy
import random
from datetime import datetime

random.seed(datetime.now())

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


class FastShapeletCandidates:
    """
    Generate shapelet candidates according to [1].
    Parameters
    ----------
    n_or_threshold : int or float
        the number of LFDPs to extract from each time series (int) according to [1]
        or the threshold to stop (float) according to [2]
    std_split : float
        the standard deviation from the mean to subdivide the time series of a class into subclasses. In [1] a
        standard deviation of 1.5 is used. Controls the number of subclasses. The smaller the more clusters.
    References
    ----------
      [1] Ji et al., „A Shapelet Selection Algorithm for Time Series Classification“.
      [2] Ji, C., Liu, S., Yang, C., Wu, L., Pan, L., & Meng, X. (2016). A piecewise linear representation method based on
      importance data points for time series data.
      2016 IEEE 20th International Conference on Computer Supported Cooperative Work in Design (CSCWD).
    """

    def __init__(self, n_lfdp_or_threshold, std_split=1.5):
        self.n_or_threshold = n_lfdp_or_threshold
        self.std_split = std_split

    def _mean_over_ts(self, l_ts):
        """
        Calculate the mean sum over a list of time series sums.
        :param l_ts: list of time series
        :type l_ts: list of array(float)
        :return: mean over the input list of time series
        :rtype: float
        """
        return sum([numpy.sum(ts) for ts in l_ts]) / len(l_ts)

    def _get_criteria_ts(self, l_ts):
        """
        Calculate and return the criteria time series.
        :param l_ts: list of time series
        :type l_ts: list of array(float)
        :return: the criteria time series
        :rtype: array(float)
        """
        mean = self._mean_over_ts(l_ts)
        i_criteria = argmin([abs(numpy.sum(ts) - mean) for ts in l_ts])
        return l_ts[i_criteria]

    def _get_ordered_ed_to_criteria(self, l_ts, criteria):
        """
        Calculate the euclidean distance to the criteria time series of a list of time series and return an ordered list
        of the time series according to the distance to the criteria time series.
        :param l_ts: list of time series
        :type l_ts: list of array(float)
        :param criteria: the criteria time series
        :type criteria: array(float)
        :return: sorted (ascending) list of time series according to the euclidean distance to the criteria time series
        :rtype: list of tuple(i, euclidean dist time series i to criteria)
        """
        return sorted([(i, numpy.linalg.norm(ts - criteria)) for i, ts in enumerate(l_ts)], key=lambda x: x[1])

    def _calc_adjacent_discrepancies(self, l_ts_dists):
        """
        Calculate the distance discrepancies between two adjacent time series.
        :param l_ts_dists: list of tuple(i, euclidean dist time series i to criteria)
        :type l_ts_dists: list of tuple(int, float)
        :return: list of adjacent discrepancies between a list of ordered time series
        :rtype: list(float)
        """
        return [abs(ts_dist[1] - l_ts_dists[i + 1][1]) for i, ts_dist in enumerate(l_ts_dists[:-1])]

    def _get_splitpoints_class_to_subclasses(self, l_adj_discr):
        """
        Calculate the split points to subdivide time series according to distance discrepancies of adjacent time series with
        respect to a criteria time series.
        :param l_adj_discr: list of adjacent discrepancies between a list of ordered time series
        :type l_adj_discr: list(float)
        :param split_factor: (standard_dev of adjacent discrepancies) * split_factor will be used as split points
        :type split_factor: float
        :return: list of indices to subdivide the time series
        :rtype: list(int)
        """
        std = numpy.std(l_adj_discr)
        return [i for i, discr in enumerate(l_adj_discr) if discr >= std * self.std_split]

    def _split_class_to_subclasses(self, l_ts_c):
        """
        Split the time series of class into subclasses by splitting the time series into subclasses according to the
        adjacent discrepancies to a criteria time series.
        :param l_ts_c: list of time series of a class
        :type l_ts_c: list of array(float)
        :param split_factor: (standard_dev of adjacent discrepancies) * split_factor will be used as split points
        :type split_factor: float
        :return: list of list of time series, whereas each second-level list refers to a subclass
        :rtype: list of list of array(float)
        """
        # get the criteria time series
        criteria_ts = self._get_criteria_ts(l_ts_c)

        l_dists = self._get_ordered_ed_to_criteria(l_ts_c, criteria_ts)
        l_adj_discr = self._calc_adjacent_discrepancies(l_dists[1:])
        split_points = self._get_splitpoints_class_to_subclasses(l_adj_discr)

        # calculate the subclasses
        subclasses_dists = []
        for i_split_start, to in enumerate(split_points[1:]):
            split_from = split_points[i_split_start]
            subclasses_dists.append(l_dists[split_from:to])

        # subclasses_dists only contains the indices to the time series in l_ts_c
        # so here, create the actual list of subclasses of time series
        subclasses_ts = [
            [l_ts_c[j]
             for j, _ in subclass]
            for subclass in subclasses_dists
        ]

        return subclasses_ts

    def _sample_ts_from_class(self, l_ts_c):
        """
        Sample time series from a class according to the method from the paper.
        :param l_ts_c: list of time series of a class
        :type l_ts_c: list of array(float)
        :param split_factor: (standard_dev of adjacent discrepancies) * split_factor will be used as split points
        :type split_factor: float
        :return: a list of samples of the subclasses, whereas we draw exactly one sample from each subclass
        :rtype: list of array(float)
        """

        subclasses = self._split_class_to_subclasses(l_ts_c)

        sample = [subclass[random.randint(0, len(subclass) - 1)] for subclass in subclasses]
        return sample

    def _get_dists_to_fitting_line(self, segment):
        """
        Calculate the distances of each time step of the input segment to a fit of start and end point and return the
        array of distances.
        :param segment: a segment of a time series
        :type segment: array(float)
        :return: array of distances to the fitting line
        :rtype: array(float)
        """
        if len(segment) < 2:
            return segment

        s = (0, segment[0])
        e = (len(segment) - 1, segment[-1])

        # calculate line from start to end segment
        slope = (s[1] - e[1]) / (s[0] - e[0])
        b = (s[0] * e[1] - e[0] * s[1]) / (s[0] - e[0])

        # generate fitting points along the line at the position of the time steps
        fitting_points = numpy.array([slope * x + b for x in range(len(segment))])
        # calculate distance of each time step to the fitting points
        dists_to_fitting_line = numpy.abs(segment - fitting_points)

        return dists_to_fitting_line

    def _get_LFDP(self, segment):
        """
        Get LFDP of a time series segment
        """
        dists_to_fitting_line = self._get_dists_to_fitting_line(segment)
        return numpy.max(dists_to_fitting_line), numpy.argmax(dists_to_fitting_line)

    def _calc_weight_of_segment(self, segment):
        """
        Calculate the weight of a segment, whereas weight is max(sum_distances, 2*max_distance) of distance to fitting line.
        :param segment: a segment of a time series
        :type segment: array(float)
        :return: the weight of the segment
        :rtype: float
        """
        dists_to_fitting_line = self._get_dists_to_fitting_line(segment)

        sum_dists = numpy.sum(dists_to_fitting_line)
        max_dist_2 = numpy.max(dists_to_fitting_line) * 2
        return max(sum_dists, max_dist_2)

    def _sort_segments_by_weight(self, l_segments):
        """
        Sort LFDP segments by weight, whereas weight is max(sum_distances, 2*max_distance) of distance to the fitting line
        from start to end of the segment.
        :param l_segments: list of time series segments
        :type l_segments:  list of array(float)
        :return: sorted (descending) list of time series segments according to their weight
        :rtype: list of array(float)
        """
        if len(l_segments) < 2:
            return l_segments

        l_segments_weights = sorted(
            [(i, self._calc_weight_of_segment(t_segment[0])) for i, t_segment in enumerate(l_segments)],
            key=lambda x: x[1], reverse=True)

        l_segments_sorted = [l_segments[i] for i, _ in l_segments_weights]
        return l_segments_sorted

    def identify_LFDPs_of_ts(self, ts):
        """
        Identify the first n LFDPs of a time series.
        :param ts: a time series
        :type ts: array(float)
        :param n: the number of LFDPs to identify if an integer or the error threshold to stop
        :type n: int or float
        :return: a list of the indices of the time steps of the first n LFDPs of the input time series
        :rtype: list(int)
        """

        if isinstance(self.n_or_threshold, int) and not 2 < self.n_or_threshold < len(ts):
            raise ValueError("Number of LFDPs must be higher than 2 and lower than the length of the time series.")
        if isinstance(self.n_or_threshold, float) and not self.n_or_threshold > 0:
            raise ValueError("The distance threshold to the fitting line must be larger than 0.")

        # start and endpoint are our first trivial LFDPs
        l_lfdp = [0, len(ts) - 1]
        n_lfdp = 2
        l_segments = [(ts, 0)]
        lfdp_dist = float('inf')
        while True:
            l_segments = self._sort_segments_by_weight(l_segments)

            # if you want to extract n LFDPs
            if isinstance(self.n_or_threshold, int) and n_lfdp >= self.n_or_threshold:
                break
            # if you want to extract LFDPs according to a threshold to the fitting line
            if isinstance(self.n_or_threshold, float) and lfdp_dist < self.n_or_threshold:
                break

            # calculate lfdp
            t_segment = l_segments[0]
            segment = t_segment[0]
            offset = t_segment[1]
            lfdp_dist, lfdp = self._get_LFDP(segment)
            l_lfdp.append(offset + lfdp)
            n_lfdp += 1

            # create new segments according to lfdp
            segment_ts_l = segment[:lfdp + 1]
            t_segment_l = (segment_ts_l, offset)
            segment_ts_r = segment[lfdp:]
            t_segment_r = (segment_ts_r, offset + lfdp)

            # update segment list
            l_segments.pop(0)
            l_segments.append(t_segment_l)
            l_segments.append(t_segment_r)

        return sorted(l_lfdp)

    def _generate_shapelet_candidates_from_LFDPs(self, ts, l_lfdp):
        """
        Generate a list of shapelet candidates from a given time series and already extracted LFDPs.
        :param ts: a time series to sample shapelet candidates from
        :type ts: array(float)
        :param l_lfdp: a list of indices of LFDPs in time series ts
        :type l_lfdp: list(int)
        :return: list of shapelet candidates (candidates have varying length)
        :rtype: list of array(float)
        """
        l_shapelet_candidates = []
        for j in range(len(l_lfdp) - 3):
            begin = l_lfdp[j]
            for i in range(j + 2, len(l_lfdp) - 1):
                end = l_lfdp[i]
                shapelet_candidate = ts[begin:end]
                l_shapelet_candidates.append(shapelet_candidate)
        return l_shapelet_candidates

    def transform(self, X):
        """
        Generate shapelet candidates according to the paper
        A Shapelet Selection Algorithm for Time Series Classification: New Directions
        :param X: a dataset of univariate time series
        :type X: array(n_time_series, length_of_time_series)
        :param n_lfdp: the number of LFDPs to extract from each time series or the threshold to stop
        :type n_lfdp: int or float
        :return: a list of shapelet candidates
        :rtype:
        """
        shapelet_candidates = []

        # sample time series
        l_ts_samples = self._sample_ts_from_class(X)

        # sample shapelet candidates
        for ts in l_ts_samples:
            lfdps = self.identify_LFDPs_of_ts(ts)
            new_candidates = self._generate_shapelet_candidates_from_LFDPs(ts, lfdps)
            shapelet_candidates += new_candidates

        return shapelet_candidates

