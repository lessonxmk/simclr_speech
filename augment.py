import numpy as np
import random
import torch


def Crop(audioData, proportion=0.5):
    '''
    Randomly select 2 points and crop the audio between the points
    audio_data: [Batch size, Time * Rate]
    proportion: (0, 1). control the max crop proportion of the audio
    '''
    batchSize = audioData.shape[0]
    audioLength = audioData.shape[1]
    cropMatrix = []
    for audio in batchSize:
        begin = np.random.randint(1, audioLength * (1 - proportion))
        end = np.random.randint(begin, begin + audioLength * proportion)
        left = np.ones(begin)
        mid = np.zeros(end - begin)
        right = np.ones(audioLength - end)
        mask = np.concatenate((left, mid, right), 0)
        cropMatrix.append(mask)
    cropMatrix = np.array(cropMatrix)
    cropMatrix = torch.from_numpy(cropMatrix)
    audioData = audioData.mul(cropMatrix)
    return audioData


def Gaussian_white_noise(audio_data, intensity=1):
    noise = torch.randn(audio_data.shape) * intensity
    audio_data += noise
    return audio_data
