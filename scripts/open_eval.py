# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:24:34 2015

@author: marchand
"""

import sys
import csv
import os
import numpy as np
import lxml.etree as etree

try:
    import jams

    JAMS_LIB = True
except:
    JAMS_LIB = False
    print('You need jams lib to create jams files (https://github.com/marl/jams.git)')
    raise Exception('You need jams lib to create jams files (https://github.com/marl/jams.git)')


def load_annotations(file_):
    ''' Read a musicdescription xml file.

    Returns a 4-column matrix whose columns are time (in sec), is_beat (1
    if the marker is a beat or a downbeat 0 in case of tatum), is_tatum
    (1 if the marker is a tatum, a beat or a downbeat), and is_measure
    (1 if the marker is a downbeat).


    Args:
        file_ (str): path of the xml file.

    Returns:
        np.array: a 4-column (time, is_beat, is_tatum, is_measure) matrix
        with each row representing a marker.
    '''
    tree = etree.parse(file_)
    data = []
    for elem in tree.iter():
        if elem.tag[-7:] == 'segment':
            a = elem.getchildren()
            if 'beat' in a[0].keys():
                b = int(a[0].get('beat'))
                tatum = int(a[0].get('tatum'))
                t = float(elem.get('time'))
                m = int(a[0].get('measure'))
                data.append([t, b, tatum, m])

    return np.asarray(data)


def swing_groundtruth_from_annot(file_):
    ''' Return a dic of groundtruth.

    Args:
        file_ (str): path of the xml file.

    Returns:
        dict: containing temporal information of markers::

            {
                'swing_median': swing ration median over the whole track,
                'swing_iqr': idem with iqr (inter-quartile-range)
                'swing_mean': idem with mean
                'swing_std': idem with std (standard deviation)

                'tempo_mean': tempo mean over the whole track
                'tempo_std': idem with std

                'beat_by_measure': a list containing the number of beat for each
                measure in the track

                'percentage_of_swing': the number of 8th-note markers over the
                number of beat markers
            }

    '''

    def iqr(x):
        return np.subtract(*np.percentile(x, [75, 25]))

    out = {}
    data = load_annotations(file_)

    # swing
    idx_swing = np.argwhere(data[:, 1] == 0)
    if len(idx_swing):
        # patch due to annotation files that ends with a swing annotation
        if idx_swing[-1] + 1 == data.shape[0]:
            idx_swing = idx_swing[:-1]
        # idem for files that begins with a swing annotation
        if idx_swing[0] == 0:
            idx_swing = idx_swing[1:]

        short_eight = data[idx_swing + 1, 0] - data[idx_swing, 0]
        long_eight = data[idx_swing, 0] - data[idx_swing - 1, 0]

        swings = long_eight / short_eight
    else:
        swings = [1.]
    out['swing_median'] = np.around(np.median(swings), decimals=3)
    out['swing_iqr'] = iqr(swings)
    out['swing_mean'] = np.around(np.mean(swings), decimals=3)
    out['swing_std'] = np.std(swings)

    # tempo
    d = np.diff(data[data[:, 1] >= 1, 0])  # select all beats : data[data[:, 1] >= 1, 0]
    out['tempo_mean'] = (60. / d).mean()
    out['tempo_std'] = (60. / d).std()

    # number of beat by measure
    d = data[data[:, 1] == 1, 3]
    a = np.argwhere(d == 1)
    out['beat_by_measure'] = np.diff(a.reshape(a.size))

    # percentage of swing
    d = data[data[:, 2] == 1, 1]
    out['percentage_of_swing'] = (d == 0).sum() / (d.sum() - 1)

    return out


def import_metadata(file_='GTZANindex.txt'):
    ''' Import title and artist from Sturm's file

    Args:
        file_ (str): path to Sturm's file
    Returns:
        dict: {audio_filename: [artist, title]}
    '''
    import re
    # test = "blues.00002.wav ::: John Lee Hooker ::: Think Twice Before You Go\n"
    re_metadata = re.compile(
        '^((?:blues|classical|country|disco|hiphop|jazz|metal|pop|reggae|rock)\.[\d]{5}\.wav) ::: (.*?) ?::: ?(.*?)\n?$')

    metadata = {}
    with open('GTZANindex.txt', 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            m = re_metadata.match(line)
            if m is not None:
                filename, artist, title = m.groups()
                metadata[filename] = [artist, title]
            else:
                raise Exception('Error parsing line: "{}"'.format(line))
    return metadata


def generate_csv_jams(folder, version_tag=''):
    '''Generates a .csv file_ containing high-level infos and jams files.

    Generates a .csv file_ containing high-level informations, given the root
    folder of annotations. This folder should contain 3 folders named 'swing',
    'no_swing' and 'ternary'. Each folder should contain a bunch of .xml
    files following the muscidescription format.

    Writes stats.csv next to these 3 folders.

    stats.csv contains a line for each .xml, and a number of columns described
    here::

        {
            'filename': audio filename,
            'tempo mean': tempo mean over the track,
            'tempo std': tempo std over the track,
            'swing ?': 'yes' if track has swing, 'no' instead,
            'swing ratio median': swing ratio median over the track,
            'swing ratio iqr': swing ratio iqr over the track,
            'swing confidence': percentage of swinged 8th-note in the track,
            'meter',
            'ternary': 'yes' if track is ternary, 'no' if not,
            'beat by measure': list of number of beat by measure,
        }

    Generates a jams file for each annotation file.
    Writes them in folder/jams/.

    Args:
        folder (str): path of the root folder of annotations.
        version_tag (str): optional, version tag to put in jams file

    '''
    try:
        os.mkdir(os.path.join(folder, 'jams'))
    except:
        pass

    metadata = import_metadata()

    with open(os.path.join(folder, 'stats.csv'), 'wb') as csvfile:
        fieldnames = ['filename', 'artist', 'title', 'tempo mean', 'tempo std', 'swing ?', 'swing ratio median',
                      'swing ratio iqr', 'swing confidence', 'meter', 'ternary ?', 'beat by measure']
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(fieldnames)

        to_write = []
        for path, subdirs, files in os.walk(folder):
            for file_ in files:
                if file_[-4:] != '.xml':
                    continue
                file_path = os.path.join(path, file_)

                filename = file_[:-4]
                swing = 'no' if 'no_swing' in path else 'yes'

                gt_dic = swing_groundtruth_from_annot(file_path)
                tempo, tempo_var = gt_dic['tempo_mean'], gt_dic['tempo_std']
                bbm = gt_dic['beat_by_measure']
                if np.unique(bbm).size == 1:
                    if 'ternary' in path:
                        meter = '{}/8'.format(bbm[0] * 3)
                    else:
                        meter = '{}/4'.format(bbm[0])
                else:
                    meter = ''

                ternary = 'yes' if 'ternary' in path else 'no'

                if swing == 'yes':
                    swing_ratio, swing_ratio_iqr = format(gt_dic['swing_median'], '.2f'), format(gt_dic['swing_iqr'],
                                                                                                 '.3f')
                    confidence = gt_dic['percentage_of_swing']
                else:
                    swing_ratio, swing_ratio_iqr, confidence = '', '', ''

                artist, title = metadata[filename]

                # csv
                to_write.append([filename, artist, title, format(tempo, '.2f'),
                                 format(tempo_var, '.2f'), format(swing, 's'), swing_ratio,
                                 swing_ratio_iqr, confidence, meter, ternary, bbm])

                # jams
                if JAMS_LIB:
                    data = load_annotations(file_path)
                    create_jams_file(filename, data, artist, title, format(tempo, '.2f'),
                                     format(tempo_var, '.2f'), format(swing, 's'), swing_ratio,
                                     swing_ratio_iqr, confidence, meter, ternary,
                                     os.path.join(folder, 'jams', filename + '.jams'),
                                     version_tag)

        # sort by filenames
        to_write.sort(key=lambda x: x[0])

        csvwriter.writerows(to_write)


def create_jams_file(filename, data, artist, title, tempo, tempo_var, swing,
                     swing_ratio, swing_ratio_iqr, confidence, meter, ternary,
                     jams_file, version_tag):
    jam = jams.JAMS()

    # ---------------------------------------------------
    # metadata
    jam.file_metadata.duration = 30.0
    jam.file_metadata.artist = artist
    jam.file_metadata.title = title
    jam.file_metadata.identifiers = {'filename': filename}

    # ---------------------------------------------------
    # annotations: beat
    ann = jams.Annotation(namespace='beat')
    for b in data[data[:, 1] >= 1, 0]:
        ann.append(time=b, duration=0.0, confidence=1, value=1)
    ann.annotation_metadata = jams.AnnotationMetadata(data_source='Manual annotations.',
                                                      annotator=jams.Curator('Ugo Marchand & Quentin Fresnel'),
                                                      corpus='GTZAN',
                                                      annotation_tools='Audioscuplt 3.3.9',
                                                      version=version_tag,
                                                      )
    ann.annotation_metadata.curator = jams.Curator('Ugo Marchand', 'ugo.marchand@ircam.fr')
    ann.sandbox = {'annotation_type': 'beat'}
    jam.annotations.append(ann)

    # ---------------------------------------------------
    # annotations: downbeat
    ann = jams.Annotation(namespace='beat')
    for b in data[data[:, 3] == 1, 0]:
        ann.append(time=b, duration=0.0, confidence=1, value=1)
    ann.annotation_metadata = jams.AnnotationMetadata(data_source='Manual annotations.',
                                                      annotator=jams.Curator('Ugo Marchand & Quentin Fresnel'),
                                                      corpus='GTZAN',
                                                      annotation_tools='Audioscuplt 3.3.9',
                                                      version=version_tag,
                                                      )
    ann.annotation_metadata.curator = jams.Curator('Ugo Marchand', 'ugo.marchand@ircam.fr')
    ann.sandbox = {'annotation_type': 'downbeat'}
    jam.annotations.append(ann)

    # ---------------------------------------------------
    # annotations: 8th-note
    if swing == 'yes':
        ann = jams.Annotation(namespace='beat')
        for b in data[data[:, 1] == 0, 0]:
            ann.append(time=b, duration=0.0, confidence=1, value=1)
        ann.annotation_metadata = jams.AnnotationMetadata(data_source='Manual annotations.',
                                                          annotator=jams.Curator('Ugo Marchand'),
                                                          corpus='GTZAN',
                                                          annotation_tools='Audioscuplt 3.3.9',
                                                          version=version_tag,
                                                          )
        ann.annotation_metadata.curator = jams.Curator('Ugo Marchand', 'ugo.marchand@ircam.fr')
        ann.sandbox = {'annotation_type': '8th-note'}
        jam.annotations.append(ann)

    # ---------------------------------------------------
    # tags manual
    tag = jams.Annotation(namespace='tag_open')
    tag.annotation_metadata = jams.AnnotationMetadata(data_source='Manual annotations.',
                                                      annotator=jams.Curator('Ugo Marchand'),
                                                      corpus='GTZAN',
                                                      annotation_tools='Audioscuplt 3.3.9',
                                                      version=version_tag,
                                                      )
    tag.annotation_metadata.curator = jams.Curator('Ugo Marchand', 'ugo.marchand@ircam.fr')
    tag.sandbox = {'swing': swing, 'ternary': ternary}
    jam.annotations.append(tag)

    # ---------------------------------------------------
    # tags automatic
    tag = jams.Annotation(namespace='tag_open')
    tag.annotation_metadata = jams.AnnotationMetadata(data_source='Automatic values.',
                                                      corpus='GTZAN',
                                                      annotation_tools='generate.py',
                                                      version=version_tag,
                                                      )
    tag.annotation_metadata.curator = jams.Curator('Ugo Marchand', 'ugo.marchand@ircam.fr')
    tag.sandbox = {
        'tempo mean': tempo,
        'tempo std': tempo_var,
        'swing ratio': swing_ratio,
        'swing ratio iqr': swing_ratio_iqr,
        'swing ratio confidence': confidence,
        'meter': meter,
    }
    jam.annotations.append(tag)

    jam.save(jams_file)


# %%
if __name__ == '__main__':
    print('generating {} and jams files (might take a long time...)'.format(os.path.join(sys.argv[1], 'stats.csv')))
    folder, version_tag = sys.argv[1]
    generate_csv_jams(folder, version_tag)
