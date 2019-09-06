import typing

from matplotlib import pylab as plt
import pandas as pd


def parse_speaker_sim(line: str) -> typing.Tuple[str, str, float]:
    """
    A line looks like
    # ori: speaker: p303, utterance: p303_233.wav, text: Mr King, who was present at yesterday's hearing, refused to comment., sim: 0.9198638200759888
    # gen: speaker: p303, utterance: p303_233.wav, text: Mr King, who was present at yesterday's hearing, refused to comment., sim: 0.8772715926170349
    :param line:
    :return:
    """
    parts = line.split(',')

    speaker = 'unknown'
    speaker_parts = parts[0].split(':')
    if 'speaker' in speaker_parts[1]:
        speaker = speaker_parts[-1].strip(' ')

    utterance = 'unknown'
    utterance_parts = parts[1].split(':')
    if 'utterance' in utterance_parts[0]:
        utterance = utterance_parts[-1].strip(' ')

    sim = 0
    sim_parts = parts[-1].split(':')
    if 'sim' in sim_parts[0]:
        sim = float(sim_parts[-1].strip(' '))

    return speaker, utterance, sim


def parse_logfile(filename: str):
    oris = []
    gens = []
    with open(filename) as f:
        for line in f:
            if line.startswith('ori'):
                oris.append(line)
            elif line.startswith('gen'):
                gens.append(line)
            else:
                pass

    # speaker => [sim1, sim2, ...]
    ori_speaker_sims = dict()  # type: typing.Dict[str, typing.Dict[str, float]]
    for ori in oris:
        speaker, utterance, sim = parse_speaker_sim(ori)
        if speaker not in ori_speaker_sims:
            ori_speaker_sims[speaker] = dict()

        ori_speaker_sims[speaker][utterance] = sim

    # speaker => [sim1, sim2, ...]
    gen_speaker_sims = dict()  # type: typing.Dict[str, typing.Dict[str,float]]
    for gen in gens:
        speaker, utterance, sim = parse_speaker_sim(gen)
        if speaker not in gen_speaker_sims:
            gen_speaker_sims[speaker] = dict()

        gen_speaker_sims[speaker][utterance] = sim

    """
    # speaker => utterance => (ori_sim, gen_sim)
    ori_gen_sims = dict()  # type: typing.Dict[str, typing.Dict[str, typing.Tuple[float, float]]]
    for speaker in ori_speaker_sims:
        if speaker not in ori_gen_sims:
            ori_gen_sims[speaker] = dict()
        for utterance in ori_speaker_sims[speaker]:
            ori_gen_sims[speaker][utterance] = (ori_speaker_sims[speaker][utterance],
                                                gen_speaker_sims[speaker][utterance])

    sim_diff = dict()  # type: typing.Dict[str, typing.Dict[str, float]]
    for speaker in ori_gen_sims:
        for utterance in ori_gen_sims[speaker]:
            sim_diff[speaker][utterance] = ori_gen_sims[speaker][utterance][0] - ori_gen_sims[speaker][utterance][1]
    """

    # convert to pandas DateFrame
    temp_dict = {
        'speaker': [],
        'utterance': [],
        'ori_sim': [],
        'gen_sim': []
    }  # type: typing.Dict[str, typing.Union[typing.List[str], typing.List[float]]]
    for speaker in ori_speaker_sims:
        for utterance in ori_speaker_sims[speaker]:
            if speaker not in gen_speaker_sims or utterance not in gen_speaker_sims[speaker]:
                continue
            temp_dict['speaker'].append(speaker)
            temp_dict['utterance'].append(utterance)
            temp_dict['ori_sim'].append(ori_speaker_sims[speaker][utterance])
            temp_dict['gen_sim'].append(gen_speaker_sims[speaker][utterance])

    return pd.DataFrame.from_dict(temp_dict)


def plot_hists(df: pd.DataFrame):
    plt.hist(df['ori_sim'], bins=20, label='ori', alpha=0.5)
    plt.hist(df['gen_sim'], bins=20, label='gen', alpha=0.5)
    plt.legend(loc='best')
    plt.plot([0.8, 0.8], [0, 100], '-')
    plt.show()


if __name__ == '__main__':
    df = parse_logfile('/Users/dalei/go/src/github.com/CorentinJ/Real-Time-Voice-Cloning/synthetic_attack_res_err.txt')
    plot_hists(df)
